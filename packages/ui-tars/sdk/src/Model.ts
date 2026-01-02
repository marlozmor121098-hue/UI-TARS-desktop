/*
 * Copyright (c) 2025 Bytedance, Inc. and its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */
import OpenAI, { type ClientOptions } from 'openai';
import {
  type ChatCompletionCreateParamsNonStreaming,
  type ChatCompletionCreateParamsBase,
  type ChatCompletionMessageParam,
} from 'openai/resources/chat/completions';
import { actionParser } from '@ui-tars/action-parser';

import { useContext } from './context/useContext';
import { Model, type InvokeParams, type InvokeOutput } from './types';

import {
  preprocessResizeImage,
  convertToOpenAIMessages,
  convertToResponseApiInput,
  isMessageImage,
} from './utils';
import { DEFAULT_FACTORS } from './constants';
import {
  UITarsModelVersion,
  MAX_PIXELS_V1_0,
  MAX_PIXELS_V1_5,
  MAX_PIXELS_DOUBAO,
} from '@ui-tars/shared/types';
import type {
  ResponseCreateParamsNonStreaming,
  ResponseInputItem,
} from 'openai/resources/responses/responses';

type OpenAIChatCompletionCreateParams = Omit<ClientOptions, 'maxRetries'> &
  Pick<
    ChatCompletionCreateParamsBase,
    'model' | 'max_tokens' | 'temperature' | 'top_p'
  >;

export interface UITarsModelConfig extends OpenAIChatCompletionCreateParams {
  /** Whether to use OpenAI Response API instead of Chat Completions API */
  useResponsesApi?: boolean;
}

export interface ThinkingVisionProModelConfig
  extends ChatCompletionCreateParamsNonStreaming {
  thinking?: {
    type: 'enabled' | 'disabled';
  };
}

export class UITarsModel extends Model {
  constructor(protected readonly modelConfig: UITarsModelConfig) {
    super();
    this.modelConfig = modelConfig;
  }

  get useResponsesApi(): boolean {
    return this.modelConfig.useResponsesApi ?? false;
  }
  private headImageContext: {
    messageIndex: number;
    responseIds: string[];
  } | null = null;

  /** [widthFactor, heightFactor] */
  get factors(): [number, number] {
    return DEFAULT_FACTORS;
  }

  get modelName(): string {
    return this.modelConfig.model ?? 'unknown';
  }

  /**
   * reset the model state
   */
  reset() {
    this.headImageContext = null;
  }

  get isGemini(): boolean {
    const { baseURL, model } = this.modelConfig;
    const modelName = (model || '').toLowerCase();
    return (
      baseURL?.includes('generativelanguage.googleapis.com') ||
      baseURL?.includes('ai.google.dev') ||
      modelName.includes('gemini') ||
      false
    );
  }

  /**
   * call real LLM / VLM Model
   * @param params
   * @param options
   * @returns
   */
  protected async invokeModelProvider(
    uiTarsVersion: UITarsModelVersion = UITarsModelVersion.V1_0,
    params: {
      messages: Array<ChatCompletionMessageParam>;
      previousResponseId?: string;
    },
    options: {
      signal?: AbortSignal;
    },
    headers?: Record<string, string>,
  ): Promise<{
    prediction: string;
    costTime?: number;
    costTokens?: number;
    responseId?: string;
  }> {
    const { logger } = useContext();
    let { messages, previousResponseId } = params;
    const {
      baseURL,
      apiKey,
      model: originalModel,
      max_tokens,
      temperature,
      top_p,
      useResponsesApi,
      ...restOptions
    } = this.modelConfig;

    // For Gemini via OpenAI compatibility, the model name should NOT include 'models/' prefix
    // if using the /openai/ endpoint, but it depends on the specific implementation.
    // However, the official documentation for the OpenAI endpoint shows names WITHOUT 'models/'.
    let model = (originalModel || 'unknown').trim().replace(/^`|`$/g, '').trim();
    if (this.isGemini) {
      if (model.startsWith('models/')) {
        model = model.replace('models/', '');
      }
    }

    // Clone messages to avoid modifying the original array
    let effectiveMessages: any[] = messages.map(msg => ({
      ...msg,
      content: Array.isArray(msg.content) ? [...msg.content] : msg.content
    }));

    // Gemini OpenAI endpoint often fails with multiple images, consecutive same-role messages, or 'system' role.
    if (this.isGemini) {
      // 0. Convert 'system' role to 'user' as some Gemini OpenAI proxies don't support 'system'
      effectiveMessages = effectiveMessages.map(msg => ({
        ...msg,
        role: msg.role === 'system' ? 'user' : msg.role
      }));

      // 1. Ensure only the most recent image is sent.
      let imageFound = false;
      for (let i = effectiveMessages.length - 1; i >= 0; i--) {
        const msg = effectiveMessages[i];
        if (Array.isArray(msg.content)) {
          msg.content = msg.content.filter(part => {
            if (part.type === 'image_url') {
              if (imageFound) return false;
              imageFound = true;
              return true;
            }
            return true;
          }) as any;
        }
      }

      // 2. Merge consecutive messages with the same role and deep clean
      const mergedMessages: any[] = [];
      for (const msg of effectiveMessages) {
        // Deep clean content: remove empty text parts and ensure image_url is valid
        let cleanContent: any = msg.content;
        if (Array.isArray(msg.content)) {
          cleanContent = msg.content.filter(part => {
            if (part.type === 'text') return part.text && part.text.trim().length > 0;
            if (part.type === 'image_url') {
              return part.image_url && (part.image_url.url || part.image_url);
            }
            return false;
          });
        } else if (typeof msg.content === 'string') {
          if (msg.content.trim().length === 0) continue;
          cleanContent = msg.content.trim();
        }

        if (!cleanContent || (Array.isArray(cleanContent) && cleanContent.length === 0)) continue;

        const lastMsg = mergedMessages[mergedMessages.length - 1];
        if (lastMsg && lastMsg.role === msg.role) {
          // Merge content
          const lastContent = Array.isArray(lastMsg.content) ? lastMsg.content : [{ type: 'text', text: String(lastMsg.content || '') }];
          const newContent = Array.isArray(cleanContent) ? cleanContent : [{ type: 'text', text: String(cleanContent || '') }];
          lastMsg.content = [...lastContent, ...newContent];
        } else {
          mergedMessages.push({
            role: msg.role,
            content: cleanContent
          });
        }
      }

      effectiveMessages = mergedMessages;

      // 3. Ensure strictly alternating roles (user, assistant, user, assistant...)
      const alternatingMessages: any[] = [];
      let expectedRole = 'user';
      
      for (let i = 0; i < mergedMessages.length; i++) {
        const msg = mergedMessages[i];
        if (msg.role === expectedRole) {
          alternatingMessages.push(msg);
          expectedRole = expectedRole === 'user' ? 'assistant' : 'user';
        } else {
          if (expectedRole === 'user') {
            // Expected user, but got assistant. Inject a user nudge.
            alternatingMessages.push({
              role: 'user',
              content: 'Continue.'
            });
            alternatingMessages.push(msg);
            expectedRole = 'user'; // After assistant comes user
          } else {
            // Expected assistant, but got user. 
            // This happens if we have User, User (already merged) or User, User that wasn't merged.
            // Since we merged above, this shouldn't happen often.
            // If it does, we can either skip or merge with previous.
            const lastMsg = alternatingMessages[alternatingMessages.length - 1];
            if (lastMsg && lastMsg.role === 'user') {
              // Merge content with previous user message
              const lastContent = Array.isArray(lastMsg.content) ? lastMsg.content : [{ type: 'text', text: String(lastMsg.content || '') }];
              const newContent = Array.isArray(msg.content) ? msg.content : [{ type: 'text', text: String(msg.content || '') }];
              lastMsg.content = [...lastContent, ...newContent];
              // expectedRole remains assistant
            } else {
              // Should not happen if logic is correct, but for safety:
              alternatingMessages.push({
                role: 'assistant',
                content: 'I will help you with that.'
              });
              alternatingMessages.push(msg);
              expectedRole = 'assistant';
            }
          }
        }
      }
      effectiveMessages = alternatingMessages;

      // 4. Final check: must start with user and end with user (for inference)
      if (effectiveMessages.length === 0) {
        effectiveMessages.push({
          role: 'user',
          content: 'Analyze the screen and provide the next action.'
        });
      } else {
        if (effectiveMessages[0].role !== 'user') {
          effectiveMessages.unshift({
            role: 'user',
            content: 'Starting interaction.'
          });
        }
        if (effectiveMessages[effectiveMessages.length - 1].role !== 'user') {
          effectiveMessages.push({
            role: 'user',
            content: 'Please provide the next action.'
          });
        }
      }
  }

    const defaultHeaders: Record<string, string> = {};
    if (this.isGemini && apiKey) {
      defaultHeaders['x-goog-api-key'] = apiKey;
    }

    // Use the provided apiKey for Authorization header. 
    // For Gemini's OpenAI-compatible endpoint, this is usually sufficient.
    const openai = new OpenAI({
      baseURL,
      apiKey: apiKey || 'dummy',
      defaultHeaders: {
        ...defaultHeaders,
        // Some Gemini proxies/endpoints prefer x-goog-api-key
        ...(this.isGemini && apiKey ? { 'x-goog-api-key': apiKey } : {}),
      },
      maxRetries: 0,
    });

    let effectiveMaxTokens = max_tokens ?? (uiTarsVersion == UITarsModelVersion.V1_5 ? 65535 : 1000);
    if (this.isGemini) {
      // Gemini 2.5 Flash supports up to 32K input tokens, but we limit output tokens.
      // 8192 is a safe common limit for Gemini output tokens.
      if (effectiveMaxTokens > 8192) {
        effectiveMaxTokens = 8192; 
      }
    }

    const createCompletionPrams: any = {
      model,
      messages: effectiveMessages,
      stream: false,
    };

    if (effectiveMaxTokens && effectiveMaxTokens > 0) {
      createCompletionPrams.max_tokens = effectiveMaxTokens;
    }

    if (!this.isGemini) {
      createCompletionPrams.temperature = temperature ?? 0;
      createCompletionPrams.top_p = top_p ?? 0.7;
    } else {
      // Gemini is extremely sensitive to parameters in its OpenAI adapter.
      // We explicitly DO NOT set temperature or top_p to avoid 400 errors.
      // Some adapters might also fail if max_tokens is set to null or zero.
      if (!createCompletionPrams.max_tokens) {
        delete createCompletionPrams.max_tokens;
      }
    }

    // Only add thinking for non-Gemini models that might support it
    const isDeepSeek = model.toLowerCase().includes('deepseek');
    if (isDeepSeek) {
      createCompletionPrams.thinking = {
        type: 'disabled',
      };
    }

    const startTime = Date.now();

    const truncatedMessages = JSON.stringify(
      createCompletionPrams.messages,
      (key, value) => {
        if (typeof value === 'string' && value.startsWith('data:image/')) {
          return value.slice(0, 50) + '...[truncated]';
        }
        return value;
      },
      2,
    );
    logger.info('[UITarsModel] Request Payload:', {
      model: createCompletionPrams.model,
      max_tokens: createCompletionPrams.max_tokens,
      temperature: createCompletionPrams.temperature,
      top_p: createCompletionPrams.top_p,
      messages: truncatedMessages,
    });

    if (this.modelConfig.useResponsesApi) {
      const lastAssistantIndex = messages.findLastIndex(
        (c) => c.role === 'assistant',
      );
      logger.info('[ResponseAPI] lastAssistantIndex: ', lastAssistantIndex);
      // incremental messages
      const inputs = convertToResponseApiInput(
        lastAssistantIndex > -1
          ? messages.slice(lastAssistantIndex + 1)
          : messages,
      );

      // find the first image message
      const headImageMessageIndex = messages.findIndex(isMessageImage);
      if (
        this.headImageContext?.responseIds.length &&
        this.headImageContext?.messageIndex !== headImageMessageIndex
      ) {
        // The image window has slid. Delete the first image message.
        logger.info(
          '[ResponseAPI] should [delete]: ',
          this.headImageContext,
          'headImageMessageIndex',
          headImageMessageIndex,
        );
        const headImageResponseId = this.headImageContext.responseIds.shift();

        if (headImageResponseId) {
          const deletedResponse = await openai.responses.delete(
            headImageResponseId,
            {
              headers,
            },
          );
          logger.info(
            '[ResponseAPI] [deletedResponse]: ',
            headImageResponseId,
            deletedResponse,
          );
        }
      }

      let result;
      let responseId = previousResponseId;
      for (const input of inputs) {
        const truncated = JSON.stringify(
          [input],
          (key, value) => {
            if (typeof value === 'string' && value.startsWith('data:image/')) {
              return value.slice(0, 50) + '...[truncated]';
            }
            return value;
          },
          2,
        );
        const responseParams: ResponseCreateParamsNonStreaming = {
          input: [input],
          model,
          temperature,
          top_p,
          stream: false,
          max_output_tokens: max_tokens,
          ...(responseId && {
            previous_response_id: responseId,
          }),
        };

        // Add thinking only if supported
        if (isDeepSeek) {
          // @ts-expect-error
          responseParams.thinking = {
            type: 'disabled',
          };
        }

        logger.info(
          '[ResponseAPI] [input]: ',
          truncated,
          'previous_response_id',
          responseParams?.previous_response_id,
          'headImageMessageIndex',
          headImageMessageIndex,
        );

        result = await openai.responses.create(responseParams, {
          ...options,
          timeout: 1000 * 30,
          headers,
        });
        logger.info('[ResponseAPI] [result]: ', result);
        responseId = result?.id;
        logger.info('[ResponseAPI] [responseId]: ', responseId);

        // head image changed
        if (responseId && isMessageImage(input)) {
          this.headImageContext = {
            messageIndex: headImageMessageIndex,
            responseIds: [
              ...(this.headImageContext?.responseIds || []),
              responseId,
            ],
          };
        }

        logger.info(
          '[ResponseAPI] [headImageContext]: ',
          this.headImageContext,
        );
      }

      return {
        prediction: result?.output_text ?? '',
        costTime: Date.now() - startTime,
        costTokens: result?.usage?.total_tokens ?? 0,
        responseId,
      };
    }

    if (this.isGemini) {
      logger.info('[UITarsModel] Gemini Request Payload:', JSON.stringify(createCompletionPrams, null, 2));
    }

    // Use Chat Completions API if not using Response API
    try {
      logger.info(`[UITarsModel] Calling OpenAI API at ${baseURL} with model ${model}`);
      const result = await openai.chat.completions.create(
        createCompletionPrams,
        {
          ...options,
          timeout: 1000 * 30,
          headers,
        },
      );

      if (this.isGemini) {
        logger.info('[UITarsModel] Gemini Response:', JSON.stringify(result, null, 2));
      }

      return {
        prediction: result.choices?.[0]?.message?.content ?? '',
        costTime: Date.now() - startTime,
        costTokens: result.usage?.total_tokens ?? 0,
      };
    } catch (error: any) {
      if (this.isGemini) {
        logger.error('[UITarsModel] Gemini API Error:', {
          status: error?.status,
          message: error?.message,
          data: error?.response?.data || error?.data,
          payload: JSON.stringify(createCompletionPrams, null, 2).substring(0, 1000) + '...'
        });
      }
      logger?.error('[UITarsModel] OpenAI API Error:', {
        status: error?.status,
        message: error?.message,
        body: error?.body,
        headers: error?.headers,
        stack: error?.stack,
        requestPayload: {
          model: createCompletionPrams.model,
          max_tokens: createCompletionPrams.max_tokens,
          messageCount: createCompletionPrams.messages.length,
        }
      });
      // Try to log the raw response if possible
      if (error?.response) {
        try {
          const rawBody = await error.response.text();
          logger?.error('[UITarsModel] Raw Error Body:', rawBody);
        } catch (e) {
          logger?.error('[UITarsModel] Could not read raw error body:', e);
        }
      }
      throw error;
    }
  }

  async invoke(params: InvokeParams): Promise<InvokeOutput> {
    const {
      conversations,
      images,
      screenContext,
      scaleFactor,
      uiTarsVersion,
      headers,
      previousResponseId,
    } = params;
    const { logger, signal } = useContext();

    logger?.info(
      `[UITarsModel] invoke: screenContext=${JSON.stringify(screenContext)}, scaleFactor=${scaleFactor}, uiTarsVersion=${uiTarsVersion}, useResponsesApi=${this.modelConfig.useResponsesApi}`,
    );

    const maxPixels =
      uiTarsVersion === UITarsModelVersion.V1_5
        ? MAX_PIXELS_V1_5
        : uiTarsVersion === UITarsModelVersion.DOUBAO_1_5_15B ||
            uiTarsVersion === UITarsModelVersion.DOUBAO_1_5_20B
          ? MAX_PIXELS_DOUBAO
          : MAX_PIXELS_V1_0;
    const compressedImages = await Promise.all(
      images.map((image) => preprocessResizeImage(image, maxPixels)),
    );

    if (this.isGemini) {
      // For Gemini, we might need to adjust the system prompt or how it's sent
      // Gemini works best when the system prompt is explicitly marked or the first message
      // UI-TARS often embeds system prompt in the first message.
    }

    const messages = convertToOpenAIMessages({
      conversations,
      images: this.isGemini ? compressedImages.slice(-1) : compressedImages,
    });

    // Log the number of messages and images for debugging
    logger.info(`[UITarsModel] invoke: messages=${messages.length}, images=${this.isGemini ? 1 : images.length}, isGemini=${this.isGemini}`);
    if (this.isGemini) {
      logger.info(`[UITarsModel] Gemini Payload: ${JSON.stringify({ messages })}`);
    }

    const startTime = Date.now();
    const result = await this.invokeModelProvider(
      uiTarsVersion,
      {
        messages,
        previousResponseId,
      },
      {
        signal,
      },
      headers,
    )
      .catch((e) => {
        logger?.error('[UITarsModel] error', e);
        throw e;
      })
      .finally(() => {
        logger?.info(`[UITarsModel cost]: ${Date.now() - startTime}ms`);
      });

    if (this.isGemini) {
      logger.info(`[UITarsModel] Gemini Response: ${JSON.stringify(result)}`);
    }

    if (!result.prediction) {
      const err = new Error();
      err.name = 'vlm response error';
      err.stack = JSON.stringify(result) ?? 'no message';
      logger?.error(err);
      throw err;
    }

    const { prediction, costTime, costTokens, responseId } = result;

    try {
      const { parsed: parsedPredictions } = actionParser({
        prediction,
        factor: this.factors,
        screenContext,
        scaleFactor,
        modelVer: uiTarsVersion,
      });
      return {
        prediction,
        parsedPredictions,
        costTime,
        costTokens,
        responseId,
      };
    } catch (error) {
      logger?.error('[UITarsModel] error', error);
      return {
        prediction,
        parsedPredictions: [],
        responseId,
      };
    }
  }
}
