# Teachers' Day feedback regression crops

Three user-provided feedback images from 2026-09-02, reduced to only the skill
icon and four option buttons. These crops contain no character names, chat,
desktop paths, or account information. Complete screenshots remain outside
the repository. The third-party game-content notice in THIRD_PARTY_NOTICES.txt
applies to these regression assets.

`manifest.json` records the literal on-screen option labels and the expected
canonical skill. In particular, the game says 堪察令 and 中药医理, while the
bundled web bank says 勘察令 and 中医药理. 以和为贵 is an unknown distractor.

Integration tests place these crops into a synthetic dialog with the existing
layout anchors. This checks the OCR/matcher/runtime result path; it is not
evidence that the original cropped screenshots contain a complete dialog.
