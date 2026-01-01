/**
 * Copyright (c) 2025 Bytedance, Inc. and its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */
module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    '@electron-toolkit/eslint-config-ts/recommended',
    '@electron-toolkit/eslint-config-prettier',
  ],
  settings: {
    react: {
      version: 'detect',
    },
  },
  overrides: [
    {
      files: [
        'packages/ui-tars/visualizer/**/*.{ts,tsx,cts,mts,js,jsx,cjs,mjs}',
      ],
      rules: {
        '@typescript-eslint/no-explicit-any': 'off',
        '@typescript-eslint/no-non-null-asserted-optional-chain': 'off',
        'no-useless-escape': 'off',
        'no-empty': 'off',
      },
    },
    {
      files: [
        '**/*.{test,spec}.{ts,tsx,cts,mts,js,jsx,cjs,mjs}',
        '**/test/**/*.{ts,tsx,cts,mts,js,jsx,cjs,mjs}',
        '**/tests/**/*.{ts,tsx,cts,mts,js,jsx,cjs,mjs}',
        '**/fixtures/**/*.{ts,tsx,cts,mts,js,jsx,cjs,mjs}',
      ],
      rules: {
        '@typescript-eslint/no-explicit-any': 'off',
      },
    },
  ],
  rules: {
    '@typescript-eslint/no-unused-vars': 'off',
    '@typescript-eslint/explicit-function-return-type': 'off',
    'react/display-name': 'off',
    '@typescript-eslint/ban-ts-comment': 'off',
    camelcase: 'off',
    // A temporary hack related to IDE not resolving correct package.json
    'import/no-extraneous-dependencies': 'off',
    'react/react-in-jsx-scope': 'off',
    'react/destructuring-assignment': 'off',
    'react/jsx-filename-extension': 'off',
    'import/extensions': 'off',
    'react/require-default-props': 'off',
    'import/no-unresolved': 'off',
    'import/no-import-module-exports': 'off',
    'no-shadow': 'off',
    'react/function-component-definition': 'off',
    'react/jsx-props-no-spreading': 'off',
    '@typescript-eslint/no-shadow': 'off',
    'class-methods-use-this': 'off',
    'import/order': 'off',
    'no-unused-vars': 'off',
    'import/prefer-default-export': 'off',
    'no-restricted-syntax': 'off',
    'no-case-declarations': 'off',
    'no-await-in-loop': 'off',
    'react/prop-types': 'off',
    '@typescript-eslint/no-namespace': 'off',
  },
};
