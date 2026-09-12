# Teachers' Day feedback regression crops

`feedback-8-*` contains only the 2026-09-11 佛法无边 diagnostic icon and option
crops. It verifies option D with the 180-degree template. The user subsequently
confirmed the original rendition was wrong, so only the corrected version is
eligible for matching. Full screenshots remain private.

Four user-provided feedback images from 2026-09-02, reduced to only the skill
icon and four option buttons. These crops contain no character names, chat,
desktop paths, or account information. Complete screenshots remain outside
the repository. The third-party game-content notice in THIRD_PARTY_NOTICES.txt
applies to these regression assets.

`manifest.json` records the literal on-screen option labels and the expected
canonical skill. In particular, the game says 堪察令 and 中药医理, while the
bundled web bank says 勘察令 and 中医药理. 以和为贵 was an unknown distractor
before the 2026-09-10 supplement update and is now a known skill.

The fourth case displays 鹰击 but the real OCR model returns 鷹击. Its
`ocr_options` preserve that observed output separately from the visible labels.
Approximate answer labels (cases 1 and 4) must remain CANDIDATE, with the fourth
case reporting 50 for the closest option and 25 for the next closest option.

Integration tests place these crops into a synthetic dialog with the existing
layout anchors. This checks the OCR/matcher/runtime result path; it is not
evidence that the original cropped screenshots contain a complete dialog.

Case 5 adds the 2026-09-09 吃茶去了 feedback. The source website icon has
a decorative border absent from the in-game icon. This real-image case must
match B at HIGH confidence without relaxing any image score threshold.

Case 6 is the 2026-09-09 11:03 UTC diagnostic crop for 牛刀小试. The official
45px asset includes a thin rim that shifts its required rendered scale beyond
the nominal template range. Test the original 69px lossless diagnostic crop
as well as the reconstructed dialog; it must beat the 吃茶去了 distractor.
# 2026-09-10 补充回归

`feedback-7-*` 仅包含“以和为贵”诊断的技能图标及四个选项裁剪，验证新增资源后选择 B，原始完整游戏截图不入库。

`feedback-9-icon.png` 是 2026-09-12 03:04:21 UTC 诊断中的“煞气诀”技能图标裁剪。单元回归使用诊断文字及裁剪回放置信度，覆盖低可信 A“鹰击”不再阻断 D“煞气决”的候选匹配、选项换序、低可信答案、文字并列与空白选项。完整游戏截图和角色信息不入库。
