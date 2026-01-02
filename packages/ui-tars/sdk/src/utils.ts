/*
 * Copyright (c) 2025 Bytedance, Inc. and its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */
import { Jimp } from 'jimp';
import { ChatCompletionMessageParam } from 'openai/resources/chat/completions';

import { IMAGE_PLACEHOLDER, MAX_IMAGE_LENGTH } from '@ui-tars/shared/constants';
import { Conversation, Message } from '@ui-tars/shared/types';
import { DEFAULT_FACTORS, type Factors } from './constants';
import {
  ResponseInput,
  ResponseInputImage,
  ResponseInputItem,
  ResponseInputText,
} from 'openai/resources/responses/responses.js';

/**
 * Parse box string to screen coordinates
 *
 *   e.g. '[0.131,0.25,0.131,0.25]' 2560x1440 -> { x: 335.36, y: 360 }
 *
 * @param boxStr box string
 * @param screenWidth screen width
 * @param screenHeight screen height
 * @param factors scaling factor, the training space of the target model.
 * @returns screen coordinates
 */
export const parseBoxToScreenCoords = ({
  boxStr,
  screenWidth,
  screenHeight,
  factors = DEFAULT_FACTORS,
}: {
  boxStr: string;
  screenWidth: number;
  screenHeight: number;
  factors?: Factors;
}) => {
  if (!boxStr) {
    return { x: null, y: null };
  }
  const coords = boxStr
    .replace('[', '')
    .replace(']', '')
    .split(',')
    .map((num) => parseFloat(num.trim()));

  const [x1, y1, x2 = x1, y2 = y1] = coords;
  const [widthFactor, heightFactor] = factors;

  return {
    x: Math.round(((x1 + x2) / 2) * screenWidth * widthFactor) / widthFactor,
    y: Math.round(((y1 + y2) / 2) * screenHeight * heightFactor) / heightFactor,
  };
};

export const processVlmParams = (
  conversations: Message[],
  images: string[],
  maxImageLength: number = MAX_IMAGE_LENGTH,
): {
  images: string[];
  conversations: Message[];
} => {
  // Check if the images array exceeds the limit
  // TODO: configurable max image length
  if (images.length > maxImageLength) {
    // Calculate the number of items to remove
    const excessCount = images.length - maxImageLength;

    // Remove excess images from the start
    images = images.slice(excessCount);

    // Remove corresponding conversations where "value" is "<image>"
    let imageCountToRemove = excessCount;
    conversations = conversations.filter((convo) => {
      if (imageCountToRemove > 0 && convo.value === IMAGE_PLACEHOLDER) {
        imageCountToRemove--;
        return false;
      }
      return true;
    });
  }

  // Return the processed result
  return { images, conversations };
};

export const toVlmModelFormat = ({
  historyMessages,
  conversations,
  systemPrompt,
}: {
  historyMessages: Message[];
  conversations: Conversation[];
  systemPrompt: string;
}): {
  conversations: Message[];
  images: string[];
} => {
  const USER_INSTRUCTION_MARKER = '## User Instruction';
  const history = formatHistoryMessages(historyMessages);
  return {
    conversations: conversations.map((conv, idx) => {
      if (idx === 0 && conv.from === 'human') {
        let newValue = '';
        if (systemPrompt.includes(USER_INSTRUCTION_MARKER)) {
          const insertIndex = systemPrompt.lastIndexOf(USER_INSTRUCTION_MARKER);
          const slicedPrefix = systemPrompt.slice(0, insertIndex);
          const slicedSuffix = systemPrompt.slice(insertIndex);
          newValue =
            slicedPrefix +
            (slicedPrefix.endsWith('\n') ? '' : '\n') +
            history +
            '\n' +
            slicedSuffix +
            (slicedSuffix.endsWith('\n') ? '' : '\n') +
            conv.value;
        } else {
          newValue = `${systemPrompt}\n${history}\n${USER_INSTRUCTION_MARKER}\n${conv.value}`;
        }
        return {
          from: conv.from,
          value: newValue,
        };
      }
      return {
        from: conv.from,
        value: conv.value,
      };
    }),
    images: conversations
      .filter(
        (conv) => conv.value === IMAGE_PLACEHOLDER && !!conv.screenshotBase64,
      )
      .map((conv) => conv.screenshotBase64!),
  };
};

export const getSummary = (prediction: string) =>
  prediction
    .replace(/Reflection:[\s\S]*?(?=Action_Summary:|Action:|$)/g, '')
    .trim();

/**
 * convert conversations to OpenAI ChatCompletionMessageParam
 * @param conversations conversations
 * @param images images
 * @returns OpenAI ChatCompletionMessageParam
 */
export const convertToOpenAIMessages = ({
  conversations,
  images,
}: {
  conversations: Message[];
  images: string[];
}): Array<ChatCompletionMessageParam> => {
  const messages: Array<ChatCompletionMessageParam> = [];
  
  // Count total occurrences of IMAGE_PLACEHOLDER across all conversation values
  let totalPlaceholders = 0;
  conversations.forEach(c => {
    const matches = (c.value || '').match(new RegExp(IMAGE_PLACEHOLDER, 'g'));
    if (matches) totalPlaceholders += matches.length;
  });

  let currentPlaceholderIndex = 0;

  conversations.forEach((conv, index) => {
    const role = conv.from === 'human' ? 'user' : 'assistant';
    
    // Gemini requires strictly alternating roles and first message must be 'user'
    if (messages.length === 0 && role === 'assistant') {
      messages.push({
        role: 'user',
        content: 'Starting interaction.',
      });
    }

    let lastMessage = messages[messages.length - 1];
    const shouldMerge = lastMessage && lastMessage.role === role;

    // Split the value by the placeholder
    const parts = (conv.value || '').split(IMAGE_PLACEHOLDER);
    
    const contentParts: any[] = [];

    parts.forEach((part, partIdx) => {
      if (part) {
        contentParts.push({ type: 'text', text: part });
      }

      // If not the last part, it means there was a placeholder here
      if (partIdx < parts.length - 1) {
        currentPlaceholderIndex++;
        const isLastImage = currentPlaceholderIndex === totalPlaceholders;
        
        // Decide which image to use. 
        // If we only have 1 image (Gemini case), only use it for the last placeholder.
        const imageToUse = images.length === 1 
          ? (isLastImage ? images[0] : null)
          : images[currentPlaceholderIndex - 1];

        if (imageToUse) {
          contentParts.push({
            type: 'image_url',
            image_url: { 
              url: `data:image/png;base64,${imageToUse}`,
            },
          });
        } else {
          const placeholderText = (index === 0 && partIdx === 0) ? 'Analyzing screen.' : '[Image omitted]';
          contentParts.push({ type: 'text', text: placeholderText });
        }
      }
    });

    if (contentParts.length === 0) {
      contentParts.push({ type: 'text', text: 'Continue.' });
    } else if (contentParts.length === 1 && contentParts[0].type === 'image_url') {
      // Gemini and some other models require text alongside images, often preferring text first
      contentParts.unshift({ type: 'text', text: 'Analyze this screen and provide the next action.' });
    }

    // Filter out empty text parts
    const filteredContentParts = contentParts.filter(part => {
      if (part.type === 'text') return part.text && part.text.trim().length > 0;
      return true;
    });

    if (shouldMerge) {
      if (Array.isArray(lastMessage.content)) {
        lastMessage.content.push(...filteredContentParts);
      } else {
        lastMessage.content = [
          { type: 'text', text: lastMessage.content as string },
          ...filteredContentParts,
        ];
      }
    } else {
      messages.push({
        role,
        content: filteredContentParts.length === 1 && filteredContentParts[0].type === 'text' 
          ? filteredContentParts[0].text 
          : filteredContentParts,
      });
    }
  });

  // Final cleanup: merge consecutive text parts in each message
  messages.forEach(msg => {
    if (Array.isArray(msg.content)) {
      const newContent: any[] = [];
      msg.content.forEach(part => {
        const lastPart = newContent[newContent.length - 1];
        if (lastPart && lastPart.type === 'text' && part.type === 'text') {
          // Add a newline if they were separate parts and don't have one
          const separator = (lastPart.text.endsWith('\n') || part.text.startsWith('\n')) ? '' : '\n';
          lastPart.text += separator + part.text;
        } else {
          newContent.push(part);
        }
      });
      msg.content = newContent;
      // If only one text part remains, convert back to string
      if (msg.content.length === 1 && msg.content[0].type === 'text') {
        msg.content = msg.content[0].text;
      }
    }
  });

  return messages;
};

export function replaceBase64Prefix(base64: string) {
  return base64.replace(/^data:image\/\w+;base64,/, '');
}

export async function preprocessResizeImage(
  image_base64: string,
  maxPixels: number,
): Promise<string> {
  try {
    const imageBuffer = Buffer.from(image_base64, 'base64');

    const image = await Jimp.read(imageBuffer);
    const { width, height } = image.bitmap;

    const currentPixels = width * height;
    if (currentPixels > maxPixels) {
      const resizeFactor = Math.sqrt(maxPixels / currentPixels);
      const newWidth = Math.floor(width * resizeFactor);
      const newHeight = Math.floor(height * resizeFactor);

      const resized = await image
        .resize({
          w: newWidth,
          h: newHeight,
        })
        .getBuffer('image/png', { quality: 60 });

      return resized.toString('base64');
    }

    const base64 = await image.getBase64('image/png', { quality: 60 });

    return replaceBase64Prefix(base64);
  } catch (error) {
    console.error('preprocessResizeImage error:', error);
    throw error;
  }
}

function formatHistoryMessages(messages: Message[]): string {
  const lastMessages = messages.slice(-30);

  const lines = lastMessages.map((msg) => {
    const role = msg.from === 'human' ? 'human' : 'assistant';
    return `${role}: ${msg.value}`;
  });

  // human: xxx, assistant: xxx.
  // const formattedLines = lines.map((line) => {
  //   if (line.startsWith('human:')) {
  //     return line + ',';
  //   } else {
  //     return line + '.';
  //   }
  // });

  return '## History Messages\n' + lines.join('\n') + '\n';
}

/**
 * convert ChatCompletionMessageParam to Response API input
 * @param messages messages
 * @returns Response API input
 */
export const convertToResponseApiInput = (
  messages: ChatCompletionMessageParam[],
): ResponseInput => {
  return messages.map((message) => {
    if (Array.isArray(message?.content) && message?.content.length > 0) {
      const content = message.content.map((item) => {
        if (item.type === 'image_url' && item.image_url?.url) {
          return {
            type: 'input_image',
            image_url: item.image_url.url,
          } as ResponseInputImage;
        }
        return item;
      });
      return {
        role: message.role,
        content,
      } as ResponseInputItem.Message;
    }

    return message as unknown as ResponseInputItem.Message;
  });
};

/**
 * check if the message is an image message
 * @param c message
 * @returns true if the message is an image message
 */
export const isMessageImage = (
  c: ChatCompletionMessageParam | ResponseInputItem,
) =>
  'role' in c &&
  c.role === 'user' &&
  Array.isArray(c.content) &&
  c.content.some(
    (item) =>
      (item.type === 'image_url' && item.image_url?.url) ||
      (item.type === 'input_image' && item.image_url),
  );
