/**
 * Copyright (c) 2025 Bytedance, Inc. and its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */
import { createRequire } from 'node:module';

import * as env from '@main/env';
import { logger } from '@main/logger';

let hasScreenRecordingPermission = false;
let hasAccessibilityPermission = false;

const require = createRequire(import.meta.url);
type MacScreenCapturePermissions =
  typeof import('@computer-use/mac-screen-capture-permissions');
type MacPermissionsApi = typeof import('@computer-use/node-mac-permissions');

const macScreenCapturePermissions: MacScreenCapturePermissions | null =
  process.platform === 'darwin'
    ? (require('@computer-use/mac-screen-capture-permissions') as MacScreenCapturePermissions)
    : null;

const macPermissions: MacPermissionsApi | null =
  process.platform === 'darwin'
    ? (require('@computer-use/node-mac-permissions') as MacPermissionsApi)
    : null;

const wrapWithWarning =
  (message, nativeFunction) =>
  (...args) => {
    console.warn(message);
    return nativeFunction(...args);
  };

const askForAccessibility = (
  permissions: MacPermissionsApi,
  nativeFunction,
  functionName,
) => {
  const accessibilityStatus = permissions.getAuthStatus('accessibility');
  logger.info('[accessibilityStatus]', accessibilityStatus);

  if (accessibilityStatus === 'authorized') {
    hasAccessibilityPermission = true;
    return nativeFunction;
  } else if (
    accessibilityStatus === 'not determined' ||
    accessibilityStatus === 'denied'
  ) {
    hasAccessibilityPermission = false;
    permissions.askForAccessibilityAccess();
    return wrapWithWarning(
      `##### WARNING! The application running this script tries to access accessibility features to execute ${functionName}! Please grant requested access and visit https://github.com/nut-tree/nut.js#macos for further information. #####`,
      nativeFunction,
    );
  }
};
const askForScreenRecording = (
  permissions: MacPermissionsApi,
  nativeFunction,
  functionName,
) => {
  const screenCaptureStatus = permissions.getAuthStatus('screen');

  if (screenCaptureStatus === 'authorized') {
    hasScreenRecordingPermission = true;
    return nativeFunction;
  } else if (
    screenCaptureStatus === 'not determined' ||
    screenCaptureStatus === 'denied'
  ) {
    hasScreenRecordingPermission = false;
    permissions.askForScreenCaptureAccess();
    return wrapWithWarning(
      `##### WARNING! The application running this script tries to screen recording features to execute ${functionName}! Please grant the requested access for further information. #####`,
      nativeFunction,
    );
  }
};

export const ensurePermissions = (): {
  screenCapture: boolean;
  accessibility: boolean;
} => {
  if (env.isE2eTest) {
    return {
      screenCapture: true,
      accessibility: true,
    };
  }

  if (process.platform !== 'darwin') {
    return {
      screenCapture: true,
      accessibility: true,
    };
  }

  if (!macScreenCapturePermissions || !macPermissions) {
    return {
      screenCapture: false,
      accessibility: false,
    };
  }

  const {
    hasPromptedForPermission,
    hasScreenCapturePermission,
    openSystemPreferences,
  } = macScreenCapturePermissions;

  const permissions = macPermissions;

  logger.info('Has asked permissions?', hasPromptedForPermission());

  hasScreenRecordingPermission = hasScreenCapturePermission();
  logger.info('Has permissions?', hasScreenRecordingPermission);
  logger.info('Has asked permissions?', hasPromptedForPermission());

  if (!hasScreenRecordingPermission) {
    openSystemPreferences();
  }

  askForAccessibility(permissions, () => {}, 'execute accessibility');
  askForScreenRecording(permissions, () => {}, 'execute screen recording');

  logger.info(
    '[ensurePermissions] hasScreenRecordingPermission',
    hasScreenRecordingPermission,
    'hasAccessibilityPermission',
    hasAccessibilityPermission,
  );

  return {
    screenCapture: hasScreenRecordingPermission,
    accessibility: hasAccessibilityPermission,
  };
};
