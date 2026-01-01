/*
 * Copyright (c) 2025 Bytedance, Inc. and its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */
import type { WebContents } from 'electron';

export type ZodSchema<TInput> = { parse: (input: unknown) => TInput };

type BivariantHandler<TArgs, TResult> = {
  bivarianceHack(args: TArgs): Promise<TResult>;
}['bivarianceHack'];

export type HandleFunction<TInput = unknown, TResult = unknown> = BivariantHandler<
  {
    context: HandleContext;
    input: TInput;
  },
  TResult
>;

export type HandleContext = { sender: WebContents | null };

export type RouterType = Record<string, { handle: HandleFunction }>;

export type ClientFromRouter<Router extends RouterType> = {
  [K in keyof Router]: Router[K]['handle'] extends (options: {
    context: HandleContext;
    input: infer P;
  }) => Promise<infer R>
    ? (input: P) => Promise<R>
    : never;
};

export type ServerFromRouter<Router extends RouterType> = {
  [K in keyof Router]: Router[K]['handle'] extends (options: {
    context: HandleContext;
    input: infer P;
  }) => Promise<infer R>
    ? (input: P) => Promise<R>
    : never;
};
