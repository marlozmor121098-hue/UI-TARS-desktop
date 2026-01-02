/**
 * Copyright (c) 2025 Bytedance, Inc. and its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */
import { OpenAI } from 'openai';
import { initIpc } from '@ui-tars/electron-ipc/main';
import { logger } from '../logger';
import { isGeminiBaseUrl, normalizeGeminiOpenAIBaseUrl } from '../utils/agent';

const t = initIpc.create();

export const settingRoute = t.router({
  checkVLMResponseApiSupport: t.procedure
    .input<{
      baseUrl: string;
      apiKey: string;
      modelName: string;
    }>()
    .handle(async ({ input }) => {
      if (isGeminiBaseUrl(input.baseUrl)) {
        return false;
      }
      try {
        const openai = new OpenAI({
          apiKey: input.apiKey,
          baseURL: input.baseUrl,
        });
        const result = await openai.responses.create({
          model: input.modelName,
          input: 'return 1+1=?',
          stream: false,
        });
        return Boolean(result?.id || result?.previous_response_id);
      } catch (e) {
        logger.warn('[checkVLMResponseApiSupport] failed:', e);
        return false;
      }
    }),
  checkModelAvailability: t.procedure
    .input<{
      baseUrl: string;
      apiKey: string;
      modelName: string;
    }>()
    .handle(async ({ input }) => {
      const isGemini =
        input.baseUrl === 'https://ai.google.dev/gemini-api/docs/live' ||
        input.modelName === 'gemini-2.5-flash-native-audio-preview';
      const baseURL = isGemini
        ? 'https://generativelanguage.googleapis.com/v1/openai/'
        : input.baseUrl;

      logger.info(
        `[checkModelAvailability] Testing connection to ${input.baseUrl} (normalized: ${baseURL}), isGemini: ${isGemini}`,
      );

      const openai = new OpenAI({
        apiKey: input.apiKey,
        baseURL,
        defaultHeaders: isGemini ? { 'x-goog-api-key': input.apiKey } : undefined,
      });

      const tryCompletion = async (
        client: OpenAI,
        modelName: string,
        label: string,
      ) => {
        const fullURL = `${client.baseURL.replace(/\/+$/, '')}/chat/completions`;
        logger.info(
          `[checkModelAvailability] Attempting ${label}: model=${modelName}, fullURL=${fullURL}`,
        );
        try {
          const completion = await client.chat.completions.create({
            model: modelName,
            messages: [{ role: 'user', content: 'return 1+1=?' }],
            stream: false,
          });
          const content =
            completion.choices?.[0]?.message?.content || completion.id || 'OK';
          logger.info(
            `[checkModelAvailability] ${label} success: ${content.substring(0, 20)}...`,
          );
          return true;
        } catch (error: any) {
          logger.warn(
            `[checkModelAvailability] ${label} failed: ${error.message} (status: ${error.status})`,
          );
          return null;
        }
      };

      try {
        if (isGemini) {
          // Variants for Gemini - Strictly use the requested model
          const versions = ['/v1'];
          const prefixes = ['models/']; // Google models need 'models/'
          const headerConfigs = [
            { 'x-goog-api-key': input.apiKey },
            { Authorization: `Bearer ${input.apiKey}` },
          ];

          const testModels = ['gemini-2.5-flash-native-audio-preview'];

          for (const version of versions) {
            let vBaseURL = baseURL;
            if (vBaseURL.includes('/v1')) {
              vBaseURL = vBaseURL.replace('/v1', version);
            } else {
              vBaseURL = vBaseURL.replace('/openai', `${version}/openai`);
            }

            for (const headers of headerConfigs) {
              const client = new OpenAI({
                apiKey: input.apiKey,
                baseURL: vBaseURL,
                defaultHeaders: headers,
                dangerouslyAllowBrowser: true,
              });

              for (const modelVariant of testModels) {
                for (const prefix of prefixes) {
                  const testModel = modelVariant.startsWith('models/')
                    ? modelVariant.replace('models/', prefix)
                    : `${prefix}${modelVariant}`;

                  const headerLabel = headers
                    ? headers['x-goog-api-key']
                      ? 'goog-hdr'
                      : 'bearer-hdr'
                    : 'default-hdr';
                  const label = `${version} + ${headerLabel} + ${prefix || 'no-prefix'} + ${testModel}`;
                  const result = await tryCompletion(client, testModel, label);
                  if (result !== null) return result;
                }
              }
            }
          }
        } else {
          // Standard OpenAI / Other VLM
          const openai = new OpenAI({
            apiKey: input.apiKey,
            baseURL,
          });
          const result = await tryCompletion(openai, input.modelName, 'initial');
          if (result !== null) return result;
        }

        throw new Error(
          'All connection attempts failed. Please check your API key, Model Name, and Base URL.',
        );
      } catch (error: any) {
        logger.error(`[checkModelAvailability] Connection failed: ${error.message}`);
        throw error;
      }
    }),
});
