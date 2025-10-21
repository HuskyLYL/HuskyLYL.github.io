import type { SiteConfig } from "../types/config";
import { fontConfig } from "./fontConfig";

// 定义站点语言
const SITE_LANG = "zh_CN"; // 语言代码，例如：'en', 'zh_CN', 'ja' 等。

export const siteConfig: SiteConfig = {
  title: "HuskyLYL Studio",
  subtitle: "The Husky Log",
  description:
    "团子的小窝",
  keywords: [
    "吕彦良",
    "HuskyLYL",
    "AI Infra",
    "博客",
    "技术博客",
    "静态博客",
  ],

  lang: SITE_LANG,

  themeColor: {
    hue: 55, // 主题色的默认色相，范围从 0 到 360。例如：红色：0，青色：200，蓝绿色：250，粉色：345
    fixed: true, // 对访问者隐藏主题色选择器
    defaultMode: "system", // 默认模式："light" 浅色，"dark" 深色，"system" 跟随系统
  },

  favicon: [
    // 留空以使用默认 favicon
    {
      src: "/assets/images/wendi.ico", // 图标文件路径
      theme: "light", // 可选，指定主题 'light' | 'dark'
      sizes: "32x32", // 可选，图标大小
    },
  ],

  // 网站Logo
  // logoIcon 支持三种类型：Astro图标库，本地图片，网络图片
  // { type: "icon", value: "material-symbols:home-pin-outline" }
  // { type: "image", value: "/assets/images/logo.webp", alt: "Firefly Logo" }
  // { type: "image", value: "https://example.com/logo.png", alt: "Firefly Logo" }
  logoIcon: {
    type: "image",
    value: "/assets/images/wendi.ico",
    alt: "🍀",
  },

  // 追番配置
  bangumi: {
    userId: "", // 在此处设置你的Bangumi用户ID
  },

  // 文章页底部的"上次编辑时间"卡片开关
  showLastModified: true,

  // OpenGraph图片功能,注意开启后要渲染很长时间，不建议本地调试的时候开启
  generateOgImages: false,

  // 页面开关配置 - 控制特定页面的访问权限
  pages: {
    anime: false, // 追番页面开关，设为false时访问/anime/会返回404
    projects: true, // 项目展示页面开关，设为false时访问/projects/会返回404
    timeline: true, // 时间线页面开关，设为false时访问/timeline/会返回404
    skills: true, // 技能页面开关，设为false时访问/skills/会返回404
  },

  // 文章列表布局配置
  postListLayout: {
    // 默认布局模式："list" 列表模式（单列布局），"grid" 网格模式（双列布局）
    defaultMode: "list",
    // 是否允许用户切换布局
    allowSwitch: false,
  },

  // 分页配置
  pagination: {
    // 每页显示的文章数量
    postsPerPage: 8,
  },

  backgroundWallpaper: {
    // 是否启用背景壁纸功能
    enable: true,
    // 壁纸模式："banner" Banner壁纸模式，"overlay" 全屏透明覆盖模式
    mode: "banner",

    // 背景图片配置
    src: {
      // 桌面背景图片
      desktop: "/assets/images/b1.jpg",
      // 移动背景图片
      mobile: "/assets/images/b1.jpg",
    },

    // 图片位置
    // 支持所有CSS object-position值，如: 'top', 'center', 'bottom', 'left top', 'right bottom', '25% 75%', '10px 20px'..
    // 如果不知道怎么配置百分百之类的配置，推荐直接使用：'center'居中，'top'顶部居中，'bottom' 底部居中，'left'左侧居中，'right'右侧居中
    position: "10% 20%",

    // Banner模式特有配置
    banner: {
      homeText: {
        // 主页显示自定义文本（全局开关）
        enable: true,
        // 主页横幅主标题
        title: "The Husky Log",
        // 主页横幅副标题
        subtitle: [
          "Hello! I’m HuskyLYL from Huazhong University of Science and Technology, sharing life & tech.",
          //"From Shattered Sky, I Free Fall",
          //"Amidst Silenced Stars, I Deep Sleep",
          //"Upon Lighted Fyrefly, I Soon Gaze",
          //"From Undreamt Night, I Thence Shine",
          //"In Finalized Morrow, I Full Bloom",
        ],
        typewriter: {
          enable: true, // 启用副标题打字机效果
          speed: 100, // 打字速度（毫秒）
          deleteSpeed: 50, // 删除速度（毫秒）
          pauseTime: 2000, // 完全显示后的暂停时间（毫秒）
        },
      },
      //credit: {
      //  enable: {
      //    desktop: true, // 桌面端显示横幅图片来源文本
      //    mobile: false, // 移动端显示横幅图片来源文本
      //  },
      //  text: {
      //    desktop: "晚晚喵", // 桌面端要显示的来源文本
      //    mobile: "Mobile Credit", // 移动端要显示的来源文本
      //  },
      //  url: {
      //    desktop: "https://www.pixiv.net/artworks/135490046", // 桌面端原始艺术品或艺术家页面的 URL 链接
       //   mobile: "", // 移动端原始艺术品或艺术家页面的 URL 链接
       // },
      //},
      navbar: {
        transparentMode: "semifull", // 导航栏透明模式："semi" 半透明加圆角，"full" 完全透明，"semifull" 动态透明
      },
      waves: {
        enable: {
          desktop: true, // 桌面端启用波浪动画效果
          mobile: true, // 移动端启用波浪动画效果
        },
      },
    },

    // 全屏透明覆盖模式特有配置
    overlay: {
      zIndex: -1, // 层级，确保壁纸在背景层
      opacity: 0.8, // 壁纸透明度
      blur: 1, // 背景模糊程度
    },
  },

  // 目录功能
  toc: {
    // 目录功能开关
    enable: true,
    // 目录深度，1-3，1 表示只显示 h1 标题，2 表示显示 h1 和 h2 标题，依此类推
    // depth在新版已弃用
    depth: 3,
  },

  // 字体配置
  // 在src/config/fontConfig.ts中配置具体字体
  font: fontConfig,
};
