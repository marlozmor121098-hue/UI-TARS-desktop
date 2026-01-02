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

    // For OpenAI-compatible endpoint, Gemini models usually DON'T want the 'models/' prefix
    // in the body, as the endpoint path already includes the versioning.
    const model = (originalModel || 'unknown').trim().replace(/^`|`$/g, '').trim()
      .replace(/^models\//, '');

    // Clone messages to avoid modifying the original array
    let effectiveMessages = messages.map(msg => ({
      ...msg,
      content: Array.isArray(msg.content) ? [...msg.content] : msg.content
    }));

    // Gemini OpenAI endpoint often fails with multiple images or consecutive same-role messages.
    if (this.isGemini) {
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

      // 2. Strict role alternation: merge consecutive messages with the same role
       const mergedMessages: any[] = [];
       for (const msg of effectiveMessages) {
         // Skip messages with truly empty content as Gemini rejects them
         if (!msg.content || (Array.isArray(msg.content) && msg.content.length === 0)) {
           continue;
         }
         
         const lastMsg = mergedMessages[mergedMessages.length - 1];
         if (lastMsg && lastMsg.role === msg.role) {
           if (typeof lastMsg.content === 'string' && typeof msg.content === 'string') {
             lastMsg.content += '\n' + (msg.content || '');
           } else {
             const lastContent = Array.isArray(lastMsg.content) ? lastMsg.content : [{ type: 'text', text: String(lastMsg.content || '') }];
             const newContent = Array.isArray(msg.content) ? msg.content : [{ type: 'text', text: String(msg.content || '') }];
             lastMsg.content = [...lastContent, ...newContent];
           }
         } else {
           mergedMessages.push(msg);
         }
       }
       effectiveMessages = mergedMessages;
       
       // 3. Ensure it doesn't end with an assistant message
       if (effectiveMessages.length > 0 && effectiveMessages[effectiveMessages.length - 1].role === 'assistant') {
         effectiveMessages.push({
           role: 'user',
           content: 'Please provide the next action based on the state.'
         });
       }
     }

    const defaultHeaders =
      this.isGemini && apiKey
        ? {
            'x-goog-api-key': apiKey,
          }
        : {};

    const openai = new OpenAI({
      baseURL,
      apiKey,
      defaultHeaders,
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
      // Some adapters also fail if these are null.
    }

    // Only add thinking for non-Gemini models that might support it
    const isDeepSeek = model.toLowerCase().includes('deepseek');
    const createCompletionPramsWithOptionalThinking: any = isDeepSeek
      ? {
          ...createCompletionPrams,
          thinking: {
            type: 'disabled',
          },
        }
      : createCompletionPrams;

    const startTime = Date.now();

    const truncatedMessages = JSON.stringify(
      messages,
      (key, value) => {
        if (typeof value === 'string' && value.startsWith('data:image/')) {
          return value.slice(0, 50) + '...[truncated]';
        }
        return value;
      },
      2,
    );
    logger.info('[UITarsModel] Request Payload:', {
      model: createCompletionPramsWithOptionalThinking.model,
      max_tokens: createCompletionPramsWithOptionalThinking.max_tokens,
      temperature: createCompletionPramsWithOptionalThinking.temperature,
      top_p: createCompletionPramsWithOptionalThinking.top_p,
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

    // Use Chat Completions API if not using Response API
    try {
      logger.info(`[UITarsModel] Calling OpenAI API at ${baseURL} with model ${model}`);
      const result = await openai.chat.completions.create(
        createCompletionPramsWithOptionalThinking,
        {
          ...options,
          timeout: 1000 * 30,
          headers,
        },
      );

      return {
        prediction: result.choices?.[0]?.message?.content ?? '',
        costTime: Date.now() - startTime,
        costTokens: result.usage?.total_tokens ?? 0,
      };
    } catch (error: any) {
      logger?.error('[UITarsModel] OpenAI API Error:', {
        status: error?.status,
        message: error?.message,
        body: error?.body,
        headers: error?.headers,
        stack: error?.stack,
      });
      // Try to log the raw response if possible
      if (error?.response) {
        try {
          const rawBody = await error.response.text();
          logger?.error('[UITarsModel] Raw Error Body:', rawBody);
        } catch (e) {}
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
