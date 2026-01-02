/*
 * Copyright (c) 2025 Bytedance, Inc. and its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */
import { createClient } from '@ui-tars/electron-ipc/renderer';
import type { Router } from '@main/ipcRoutes';

export const api = createClient<Router>({
  ipcInvoke: (channel: string, ...args: unknown[]) => {
    if (!window.electron?.ipcRenderer) {
      console.error('window.electron.ipcRenderer is not available');
      return Promise.reject(new Error('IPC renderer not available'));
    }
    return window.electron.ipcRenderer.invoke(channel, ...args);
  },
});
