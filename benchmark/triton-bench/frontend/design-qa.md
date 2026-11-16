# 首版视觉与交互验收

final result: passed

2026-11-13。范围为用户选择的第二版视觉方向及其实际接口驱动实现，不将生成稿中的示意数据当作评测证据。

## 对照目标与证据

- 源稿：`/Users/mac/.codex/generated_images/01a1d055-cd45-7380-8f3f-d858c3631e4b/exec-d960f677-2a8d-4f29-8d72-e15907a33f77.png`，1487×1058 像素；归档为 `runs/frontend-foundation-20261113/browser-evidence/reference-selected.png`。
- 实现：`http://127.0.0.1:5174/submit`，已上传 Abs 源码包、READY、20 case、smoke；真实任务页为 `/workspace/jobs/b341987d818c441f98df828ef7f2781e`。
- 最终桌面截图：`/tmp/triton-eval-ui-20261113/final-submit-desktop.png`，1440×1024；CSS viewport 1440×1024、deviceScaleFactor=1。
- 最终窄屏截图：`/tmp/triton-eval-ui-20261113/final-submit-mobile.png`，CSS viewport 390×844、deviceScaleFactor=1，fullPage 长截图。
- 最终任务截图：同目录 `final-job-desktop.png`、`final-job-mobile.png`、`final-artifacts-desktop.png`。
- 全图联合对照：同目录 `final-comparison.jpg`，2880×1024，左源稿、右实现。源稿按目标大小归一化至 1440×1024；生成尺寸比例差约 0.06%，不据此声称逐像素匹配。
- 局部联合对照：同目录 `final-comparison-focus.jpg`，1880×880；两侧各裁取 x=470、y=130、宽940、高880 的表单区域，复核标题、文件、选择控件、说明和确认按钮。
- 原始截图、联合对照和 `final-browser-checks.json` 均已归档至 `runs/frontend-foundation-20261113/browser-evidence/`。

最终预览处于 Worker disabled，源稿示意处于 Worker enabled，因此状态标签、按钮禁用色和说明有意不同；在真实远端 Worker 启用时已通过页面点击创建并完成一次真机评测，证据为 `submit-remote-ready.jpg` 和 browser-remote-result.json。真实文件名、完整 corpus 版本、规则与验收文案替代示意文本，属于必要的产品内容差异。

## 检查与修正历史

1. 初次 IAB 对照发现 [P2] 高级设置挤占主配置区，确认按钮位置偏低。证据为 comparison-initial.jpg；已将高级设置移至确认区之后，comparison-fixed.jpg 及最终联合图确认主按钮在桌面首屏可见。
2. 窄屏初次截图发现 [P2] 最右侧文档导航被裁切。已缩小导航项内边距和图标/文字间距；最终 Playwright 实测五个导航均位于 390px 视口内，文档项 x=317、宽57，页面 scrollWidth=clientWidth=390。
3. 联合图发现 [P2] 主操作按钮和标识的比例偏小。已调整按钮最小宽190px、高44px与字号，以及桌面标识框比例；最终联合图复核通过。移动端主操作仍为整行。
4. 对比度检查后加深主按钮填充、选中导航文字和侧栏次要文字。最终基础配色的文字对比度分别为 4.599、5.164、4.879；禁用按钮按禁用状态单独呈现，不把其透明度当作正常操作配色。
5. 会话中断后原 IAB 控制服务不可用，报 `Local browser access is unavailable for this chat`；用户明确允许使用本机 Playwright。最终复核使用独立临时 Chrome 配置，仅允许 localhost 页面请求，不读取现有登录资料。此切换未补跑 NPU。

## 五项视觉检查

| 检查面 | 结论 |
| --- | --- |
| 字体与排版 | 使用本机中文字体与等宽标识字体，保留标题/小节/说明层级；文本可读且无碰撞。正文主要14–16px，比生成稿部分文字更紧凑，列为 P3 光学差异，不声称像素级一致。 |
| 间距与布局 | 保留浅色侧栏、纵向步骤、开放表单、轻分隔和底部确认条；桌面主控件可见，窄屏改单列，任务宽表在自身容器内滚动。 |
| 颜色与状态 | 暖灰/浅棕表面、橙色操作与选中态、绿色通过；为了可读性加深部分文字/按钮色，错误/等待/空值有明确文本。 |
| 图像与图标 | 标识为从选定稿生成的透明 PNG，已检查实际渲染，无缺图或占位；常规图标使用 Phosphor，无手绘 SVG/CSS 素材替代。 |
| 内容与交互 | 文案与真实 API 相符；原始 ZIP、201+REJECTED、档位口径、进度/评分/释放分离、无性能成绩与未认证说明均可见。 |

## 浏览器结果

最终 Playwright **17 项检查通过**：页面身份与非空、真实上传、未启用 Worker 的运行门禁、选择/档位联动、拒收、基础文字对比度、窄屏导航及无横向溢出、评测集搜索、任务搜索与跳转、真实分数/资源释放/用例筛选、原任务刷新、窄屏任务布局、完成任务不提供补取、报告实际下载与哈希匹配、无控制台/运行时错误或外部请求、未创建新的评测任务。

`final-browser-checks.json` 记录控制台错误0、pageErrors=0、外部请求0。报告下载为13026字节，SHA-256 与页面所列元数据一致。框架正常渲染、无错误遮罩；真实浏览器操作和截图构成验收证据，构建成功仅是辅助检查。

此前 IAB 真机闭环结果：Abs smoke 20 PASS、功能主分100、严格通过率100%、29项产物核验通过、Runner只启动一次、最终SHUTDOWN。另在独立 disabled 服务中验证了队列取消后 CANCELLED 与事件终态关闭。

## 实施检查单与剩余范围

- [x] 打开源稿、实际渲染及同输入联合对照，检查整图与局部。
- [x] 修正上述 P2 后重新捕获与复核；当前无未解决 P0/P1/P2。
- [x] 桌面与390px窄屏的核心交互、控制台、实际下载及最终数据已检查。
- [x] 保留可运行的本机入口与真实结果库；没有部署网站。
- [ ] P3：生成稿与系统字体的光学字重、次要文字密度可在下一轮视觉润色时进一步贴近。
- [ ] Safari/Firefox、完整键盘遍历与200%文字缩放尚未穷尽验证；不把首版验收扩展为跨浏览器无障碍认证。

标准性能、全量用例和公开榜单仍按独立验收处理，详见前端实施记录。
