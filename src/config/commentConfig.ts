import type { CommentConfig } from "../types/config";

export const commentConfig: CommentConfig = {
  enable: true , // 启用评论功能。当设置为 false 时，评论组件将不会显示在文章区域。
  twikoo: {
    envId: "huskylyl.netlify.app",
    lang: "en", // 设置 Twikoo 评论系统语言为英文
  },
};
