/*
 * Copyright (c) 2025 Bytedance, Inc. and its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { platform } from 'os';
import { ensurePermissions } from './systemPermissions';
import * as env from '@main/env';

// Mock the dependencies
vi.mock('@computer-use/mac-screen-capture-permissions', () => ({
  hasPromptedForPermission: vi.fn(),
  hasScreenCapturePermission: vi.fn(),
  openSystemPreferences: vi.fn(),
}));

vi.mock('@computer-use/node-mac-permissions', () => ({
  getAuthStatus: vi.fn(),
  askForAccessibilityAccess: vi.fn(),
  askForScreenCaptureAccess: vi.fn(),
}));
vi.mock('@main/env');
vi.mock('@main/logger');

(platform() === 'darwin' ? describe : describe.skip)(
  'systemPermissions',
  () => {
    let macScreenCapturePermissions: typeof import('@computer-use/mac-screen-capture-permissions');
    let permissions: typeof import('@computer-use/node-mac-permissions');

    beforeEach(async () => {
      vi.resetAllMocks();
      vi.mocked(env).isE2eTest = false;
      macScreenCapturePermissions = await import(
        '@computer-use/mac-screen-capture-permissions'
      );
      permissions = await import('@computer-use/node-mac-permissions');
    });

    afterEach(() => {
      vi.clearAllMocks();
    });

    it('should return true for both permissions in E2E test environment', () => {
      vi.mocked(env).isE2eTest = true;

      const result = ensurePermissions();

      expect(result).toEqual({
        screenCapture: true,
        accessibility: true,
      });
    });

    it('should handle when screen capture permission is granted', () => {
      vi.mocked(
        macScreenCapturePermissions.hasScreenCapturePermission,
      ).mockReturnValue(true);
      vi.mocked(permissions.getAuthStatus).mockImplementation((type) => {
        if (type === 'accessibility') return 'denied';
        if (type === 'screen') return 'authorized';
        return 'denied';
      });

      const result = ensurePermissions();

      expect(result.screenCapture).toBe(true);
      expect(
        macScreenCapturePermissions.hasPromptedForPermission,
      ).toHaveBeenCalled();
      expect(permissions.getAuthStatus).toHaveBeenCalledWith('accessibility');
    });

    it('should request permissions when not granted', () => {
      vi.mocked(
        macScreenCapturePermissions.hasScreenCapturePermission,
      ).mockReturnValue(false);
      vi.mocked(permissions.getAuthStatus).mockImplementation((type) => {
        if (type === 'accessibility') return 'denied';
        if (type === 'screen') return 'denied';
        return 'denied';
      });

      const result = ensurePermissions();

      expect(result.screenCapture).toBe(false);
      expect(result.accessibility).toBe(false);
      expect(
        macScreenCapturePermissions.openSystemPreferences,
      ).toHaveBeenCalled();
      expect(permissions.askForAccessibilityAccess).toHaveBeenCalled();
      expect(permissions.askForScreenCaptureAccess).toHaveBeenCalled();
    });

    it('should handle when accessibility permission is granted', () => {
      vi.mocked(
        macScreenCapturePermissions.hasScreenCapturePermission,
      ).mockReturnValue(false);
      vi.mocked(permissions.getAuthStatus).mockImplementation((type) => {
        if (type === 'accessibility') return 'authorized';
        if (type === 'screen') return 'denied';
        return 'denied';
      });

      const result = ensurePermissions();

      expect(result.accessibility).toBe(true);
      expect(result.screenCapture).toBe(false);
      expect(permissions.getAuthStatus).toHaveBeenCalledWith('accessibility');
      expect(permissions.getAuthStatus).toHaveBeenCalledWith('screen');
    });

    it('should return true when both permissions are already granted', () => {
      vi.mocked(
        macScreenCapturePermissions.hasScreenCapturePermission,
      ).mockReturnValue(true);
      vi.mocked(permissions.getAuthStatus).mockImplementation((type) => {
        if (type === 'accessibility') return 'authorized';
        if (type === 'screen') return 'authorized';
        return 'denied';
      });

      const result = ensurePermissions();

      expect(result).toEqual({
        screenCapture: true,
        accessibility: true,
      });
      expect(
        macScreenCapturePermissions.hasPromptedForPermission,
      ).toHaveBeenCalled();
      expect(permissions.getAuthStatus).toHaveBeenCalledWith('accessibility');
      expect(permissions.getAuthStatus).toHaveBeenCalledWith('screen');
    });
  },
);
