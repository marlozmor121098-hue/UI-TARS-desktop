import { UITarsModelVersion } from '@ui-tars/shared/constants';
import {
  Operator,
  SearchEngineForSettings,
  VLMProviderV2,
} from '../store/types';
import {
  getSystemPrompt,
  getSystemPromptDoubao_15_15B,
  getSystemPromptDoubao_15_20B,
  getSystemPromptV1_5,
} from '../agent/prompts';
import {
  closeScreenMarker,
  hideScreenWaterFlow,
  hideWidgetWindow,
  showScreenWaterFlow,
  showWidgetWindow,
} from '../window/ScreenMarker';
import { hideMainWindow, showMainWindow } from '../window';
import { SearchEngine } from '@ui-tars/operator-browser';

export const isGeminiBaseUrl = (baseUrl: string) =>
  baseUrl.includes('generativelanguage.googleapis.com') ||
  baseUrl.includes('ai.google.dev');

export const normalizeGeminiModelName = (modelName: string) => {
  return modelName.startsWith('models/') ? modelName : `models/${modelName}`;
};

export const normalizeGeminiOpenAIBaseUrl = (baseUrl: string) => {
  let normalized = baseUrl.trim().replace(/^`|`$/g, '').trim();
  
  // Handle documentation URLs that users might accidentally paste
  if (normalized.includes('ai.google.dev') || normalized.includes('google.dev/gemini-api')) {
    return 'https://generativelanguage.googleapis.com/v1beta/openai/';
  }

  if (normalized.includes('generativelanguage.googleapis.com')) {
    // Force v1beta for better OpenAI compatibility as per Google docs
    if (normalized.includes('/v1/')) {
      normalized = normalized.replace('/v1/', '/v1beta/');
    } else if (!normalized.includes('/v1beta/')) {
      // If no version specified, try to inject v1beta
      if (normalized.endsWith('/openai') || normalized.endsWith('/openai/')) {
        normalized = normalized.replace('/openai', '/v1beta/openai');
      }
    }
  }
  return normalized.endsWith('/') ? normalized : `${normalized}/`;
};

export const getModelVersion = (
  provider: VLMProviderV2 | undefined,
): UITarsModelVersion => {
  switch (provider) {
    case VLMProviderV2.ui_tars_1_5:
      return UITarsModelVersion.V1_5;
    case VLMProviderV2.ui_tars_1_0:
      return UITarsModelVersion.V1_0;
    case VLMProviderV2.doubao_1_5:
      return UITarsModelVersion.DOUBAO_1_5_15B;
    case VLMProviderV2.doubao_1_5_vl:
      return UITarsModelVersion.DOUBAO_1_5_20B;
    default:
      return UITarsModelVersion.V1_0;
  }
};

export const getSpByModelVersion = (
  modelVersion: UITarsModelVersion,
  language: 'zh' | 'en',
  operatorType: 'browser' | 'computer',
) => {
  switch (modelVersion) {
    case UITarsModelVersion.DOUBAO_1_5_20B:
      return getSystemPromptDoubao_15_20B(language, operatorType);
    case UITarsModelVersion.DOUBAO_1_5_15B:
      return getSystemPromptDoubao_15_15B(language);
    case UITarsModelVersion.V1_5:
      return getSystemPromptV1_5(language, 'normal');
    default:
      return getSystemPrompt(language, operatorType);
  }
};

export const getLocalBrowserSearchEngine = (
  engine?: SearchEngineForSettings,
) => {
  return (engine || SearchEngineForSettings.GOOGLE) as unknown as SearchEngine;
};

export const beforeAgentRun = async (operator: Operator) => {
  switch (operator) {
    case Operator.RemoteComputer:
      break;
    case Operator.RemoteBrowser:
      break;
    case Operator.LocalComputer:
      showWidgetWindow();
      showScreenWaterFlow();
      hideMainWindow();
      break;
    case Operator.LocalBrowser:
      hideMainWindow();
      showWidgetWindow();
      break;
    default:
      break;
  }
};

export const afterAgentRun = (operator: Operator) => {
  switch (operator) {
    case Operator.RemoteComputer:
      break;
    case Operator.RemoteBrowser:
      break;
    case Operator.LocalComputer:
      hideWidgetWindow();
      closeScreenMarker();
      hideScreenWaterFlow();
      showMainWindow();
      break;
    case Operator.LocalBrowser:
      hideWidgetWindow();
      showMainWindow();
      break;
    default:
      break;
  }
};
