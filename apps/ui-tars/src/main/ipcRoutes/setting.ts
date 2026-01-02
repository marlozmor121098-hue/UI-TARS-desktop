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
        input.baseUrl.includes('generativelanguage.googleapis.com') ||
        input.baseUrl.includes('ai.google.dev') ||
        input.modelName.toLowerCase().includes('gemini');
      const baseURL = isGemini
          ? normalizeGeminiOpenAIBaseUrl(input.baseUrl)
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
        // 1. Try exact user configuration first
        const initialResult = await tryCompletion(openai, input.modelName, 'user-config');
        if (initialResult !== null) return initialResult;

        if (isGemini) {
          // 2. If it's Gemini and user config failed, try common variants
          const versions = ['/v1beta', '/v1']; // Prefer v1beta
          const prefixes = ['models/', '']; 
          const headerConfigs = [
            { 'x-goog-api-key': input.apiKey },
            { Authorization: `Bearer ${input.apiKey}` },
          ];

          const testModels = [
            input.modelName,
            'gemini-2.5-flash',
            'gemini-2.5-flash-native-audio-preview',
          ];

          for (const version of versions) {
            let vBaseURL = baseURL;
            // Robust version replacement
            if (vBaseURL.includes('/v1beta/openai')) {
              vBaseURL = vBaseURL.replace('/v1beta/openai', `${version}/openai`);
            } else if (vBaseURL.includes('/v1/openai')) {
              vBaseURL = vBaseURL.replace('/v1/openai', `${version}/openai`);
            } else if (vBaseURL.includes('/v1beta')) {
              vBaseURL = vBaseURL.replace('/v1beta', version);
            } else if (vBaseURL.includes('/v1')) {
              vBaseURL = vBaseURL.replace('/v1', version);
            }

            for (const headers of headerConfigs) {
              const client = new OpenAI({
                apiKey: input.apiKey,
                baseURL: vBaseURL,
                defaultHeaders: headers,
                dangerouslyAllowBrowser: true,
              });

              for (const modelVariant of testModels) {
                if (!modelVariant) continue;
                for (const prefix of prefixes) {
                  const testModel = (prefix && !modelVariant.startsWith(prefix)) 
                    ? `${prefix}${modelVariant}` 
                    : modelVariant;

                  const headerLabel = headers['x-goog-api-key'] ? 'goog-hdr' : 'bearer-hdr';
                  const label = `${version} + ${headerLabel} + ${prefix || 'no-prefix'} + ${testModel}`;
                  const result = await tryCompletion(client, testModel, label);
                  if (result !== null) return result;
                }
              }
            }
          }
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
