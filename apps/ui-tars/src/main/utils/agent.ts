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
  baseUrl.includes('google.com/v1beta/openai') ||
  baseUrl.includes('google.com/v1/openai') ||
  baseUrl.includes('google.com/v1beta') ||
  baseUrl.includes('google.com/v1');

export const normalizeGeminiModelName = (modelName: string) => {
  if (
    (modelName.startsWith('gemini-') || modelName.startsWith('learnlm-')) &&
    !modelName.startsWith('models/')
  ) {
    return `models/${modelName}`;
  }
  return modelName;
};

export const normalizeGeminiOpenAIBaseUrl = (baseUrl: string) => {
  try {
    const url = new URL(baseUrl);
    let path = url.pathname.replace(/\/+$/, '');

    // Remove /chat/completions or /completions if the user accidentally included them
    path = path.replace(/\/(chat\/)?completions$/, '');

    if (!path || path === '/') {
      path = '/v1beta/openai';
    }

    if (!path.includes('/openai')) {
      if (path.endsWith('/v1beta')) {
        path = '/v1beta/openai';
      } else if (path.endsWith('/v1')) {
        path = '/v1/openai';
      } else {
        // Default to v1beta/openai if no version or openai path is present
        path = '/v1beta/openai';
      }
    }

    url.pathname = path;
    // We return with a trailing slash as per Google documentation for some SDKs,
    // although OpenAI SDK usually works without it too.
    return url.toString().replace(/\/+$/, '') + '/';
  } catch {
    return 'https://generativelanguage.googleapis.com/v1beta/openai/';
  }
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
      return getSystemPrompt(language);
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
