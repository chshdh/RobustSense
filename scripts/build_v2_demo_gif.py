"""用 V2 真实实验产物生成约 45 秒的中文讲解动图。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1280
HEIGHT = 720
BACKGROUND = "#F7F9FC"
INK = "#172033"
MUTED = "#596579"
BLUE = "#2F6FED"
GREEN = "#1E9E68"
ORANGE = "#F08A35"
RED = "#D84A4A"
PURPLE = "#7257D9"
CARD = "#FFFFFF"
MODELS = ("B4", "B5", "P", "P2")
MODEL_COLORS = {"B4": "#7A879C", "B5": BLUE, "P": ORANGE, "P2": PURPLE}


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_font(bold: bool = False) -> Path:
    windows_fonts = Path("C:/Windows/Fonts")
    names = ("msyhbd.ttc", "simhei.ttf") if bold else ("msyh.ttc", "simsun.ttc")
    for name in names:
        candidate = windows_fonts / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到可显示中文的 Windows 字体")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(find_font(bold)), size=size)


def base_frame(step: int, title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((38, 28, 1242, 692), radius=26, fill=CARD, outline="#DDE3EE", width=2)
    draw.text((72, 58), "RobustSense V2 · 面试演示", font=font(22, bold=True), fill=BLUE)
    draw.text((72, 102), title, font=font(42, bold=True), fill=INK)
    draw.text((74, 164), subtitle, font=font(21), fill=MUTED)
    draw.text((1142, 61), f"{step}/7", font=font(22, bold=True), fill=MUTED)
    draw.rounded_rectangle((72, 646, 1208, 658), radius=6, fill="#E6EBF4")
    draw.rounded_rectangle((72, 646, 72 + int(1136 * step / 7), 658), radius=6, fill=BLUE)
    return image, draw


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    value: str,
    color: str = BLUE,
    note: str | None = None,
) -> None:
    draw.rounded_rectangle(box, radius=20, fill="#F9FBFF", outline="#DCE4F1", width=2)
    x1, y1, _, _ = box
    draw.text((x1 + 24, y1 + 20), title, font=font(19, bold=True), fill=MUTED)
    draw.text((x1 + 24, y1 + 58), value, font=font(34, bold=True), fill=color)
    if note:
        draw.text((x1 + 24, y1 + 108), note, font=font(16), fill=MUTED)


def bar_chart(
    draw: ImageDraw.ImageDraw,
    values: dict[str, float],
    box: tuple[int, int, int, int],
    minimum: float,
    maximum: float,
) -> None:
    x1, y1, x2, y2 = box
    width = x2 - x1
    row_height = (y2 - y1) // len(MODELS)
    for index, model in enumerate(MODELS):
        y = y1 + index * row_height
        value = values[model]
        fraction = max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
        draw.text((x1, y + 5), model, font=font(21, bold=True), fill=MODEL_COLORS[model])
        draw.rounded_rectangle((x1 + 70, y + 5, x2, y + 35), radius=12, fill="#E8EDF5")
        draw.rounded_rectangle(
            (x1 + 70, y + 5, x1 + 70 + int((width - 70) * fraction), y + 35),
            radius=12,
            fill=MODEL_COLORS[model],
        )
        draw.text((x2 - 104, y + 8), f"{value:.4f}", font=font(17, bold=True), fill=INK)


def mean_by_model(rows: list[dict[str, str]], *, phase: str | None = None) -> dict[str, float]:
    result: dict[str, float] = {}
    for model in MODELS:
        values = [
            float(row["macro_f1"])
            for row in rows
            if row["model_id"] == model and (phase is None or row.get("phase") == phase)
        ]
        if not values:
            raise ValueError(f"{model} 没有可用的聚合值")
        result[model] = sum(values) / len(values)
    return result


def make_frames(root: Path) -> list[Image.Image]:
    acceptance = load_json(root / "reports/v2/phase_v2_5_acceptance.json")
    bootstrap = load_json(root / "reports/v2/phase_v2_4_natural_bootstrap.json")
    core_rows = read_rows(root / "reports/v2/phase_v2_4_core_summary.csv")
    mask_rows = read_rows(root / "reports/v2/phase_v2_4_mask_summary.csv")
    mixed_rows = read_rows(root / "reports/v2/phase_v2_4_mixed_summary.csv")
    persistent_rows = read_rows(root / "reports/v2/phase_v2_4_persistent_summary.csv")

    core = {row["model_id"]: float(row["natural_macro_f1_mean"]) for row in core_rows}
    mask = mean_by_model(mask_rows)
    mixed = mean_by_model(mixed_rows)
    persistent = mean_by_model(persistent_rows, phase="fault")
    p2 = acceptance["p2"]

    frames: list[Image.Image] = []

    image, draw = base_frame(1, "问题边界", "离线预计算特征回放，不伪装成实时采集或物理故障台架")
    card(draw, (74, 232, 410, 420), "任务", "多标签分类", GREEN, "sigmoid 输出；未知标签不计损失")
    card(
        draw, (472, 232, 808, 420), "输入", "六类模态", BLUE, "加速度、陀螺仪、位置、音频、手机状态"
    )
    card(
        draw,
        (870, 232, 1206, 420),
        "研究问题",
        "缺失与污染",
        ORANGE,
        "输入不可靠时，模型能否识别并谨慎融合",
    )
    draw.rounded_rectangle((74, 470, 1206, 590), radius=20, fill="#FFF7E8", outline="#F3C66D")
    draw.text((104, 493), "边界声明", font=font(22, bold=True), fill=ORANGE)
    draw.text(
        (104, 535),
        "人工故障发生在标准化特征空间；它不等同于真实硬件损坏。",
        font=font(25),
        fill=INK,
    )
    frames.append(image)

    image, draw = base_frame(
        2, "可审计的同样本对比", "四个模型读取同一官方测试切分，没有随机预测回退"
    )
    card(draw, (74, 226, 340, 382), "fold / seed", "0 / 29", BLUE)
    card(draw, (370, 226, 636, 382), "测试样本", "#0", GREEN)
    card(draw, (666, 226, 932, 382), "模型", "B4 · B5 · P · P2", PURPLE)
    card(draw, (962, 226, 1206, 382), "验收", str(acceptance["status"]), GREEN)
    draw.text((76, 430), "同一测试 split 哈希", font=font(20, bold=True), fill=MUTED)
    draw.rounded_rectangle((74, 468, 1206, 528), radius=14, fill="#F0F4FA")
    draw.text((94, 484), str(acceptance["same_test_split_sha256"]), font=font(18), fill=INK)
    draw.text(
        (76, 562),
        "任何 checkpoint、阈值或产物合同缺失都会停止运行。",
        font=font(24, bold=True),
        fill=RED,
    )
    frames.append(image)

    image, draw = base_frame(
        3, "核心结果：没有夸大 P2", "自然缺失 Macro-F1，五折 × 三种子；B5 的均值最高"
    )
    bar_chart(draw, core, (92, 232, 760, 506), minimum=0.55, maximum=0.585)
    draw.rounded_rectangle((810, 226, 1198, 510), radius=20, fill="#F8F6FF", outline="#DCD5F8")
    draw.text((842, 252), "P2 − P 配对 Bootstrap", font=font(22, bold=True), fill=PURPLE)
    draw.text(
        (842, 314),
        f"均值  {float(bootstrap['mean_delta_p2_minus_p']):+.6f}",
        font=font(25, bold=True),
        fill=INK,
    )
    draw.text(
        (842, 366),
        f"95% 区间\n[{float(bootstrap['ci_2_5']):+.6f},\n {float(bootstrap['ci_97_5']):+.6f}]",
        font=font(23),
        fill=INK,
        spacing=10,
    )
    draw.text(
        (92, 548),
        "结论：区间跨 0，不能声称 P2 在自然缺失上稳定全面优于 P。",
        font=font(23, bold=True),
        fill=RED,
    )
    frames.append(image)

    image, draw = base_frame(
        4,
        "P2 的核心：把“任务偏好”与“输入可靠度”分开",
        "五维质量 → reliability；门控网络 → utility；两者共同决定最终权重",
    )
    labels = ("输入与可用掩码", "五维质量", "reliability", "utility", "最终权重", "多标签预测")
    colors = (BLUE, ORANGE, GREEN, PURPLE, BLUE, GREEN)
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        x = 76 + index * 188
        draw.rounded_rectangle(
            (x, 250, x + 154, 340), radius=18, fill="#F7F9FD", outline=color, width=3
        )
        draw.text((x + 14, 276), label, font=font(18, bold=True), fill=color)
        if index < len(labels) - 1:
            draw.line((x + 156, 295, x + 182, 295), fill=MUTED, width=4)
            draw.polygon(((x + 182, 295), (x + 171, 287), (x + 171, 303)), fill=MUTED)
    card(
        draw,
        (154, 418, 506, 568),
        "验收故障场景系统可靠度",
        f"{float(p2['system_reliability']):.4f}",
        GREEN,
    )
    card(
        draw,
        (560, 418, 912, 568),
        "验证集冻结拒绝阈值",
        f"{float(p2['abstention_threshold']):.4f}",
        ORANGE,
    )
    card(draw, (966, 418, 1206, 568), "决定", str(p2["acceptance"]), GREEN)
    frames.append(image)

    image, draw = base_frame(5, "扩展压力评估", "63-mask、混合故障、持续故障分别回答不同问题")
    metrics = (("63-mask", mask), ("混合故障", mixed), ("持续故障中", persistent))
    for column, (label, values) in enumerate(metrics):
        left = 74 + column * 384
        draw.rounded_rectangle(
            (left, 226, left + 350, 550), radius=20, fill="#FAFBFE", outline="#DDE4EF"
        )
        draw.text((left + 24, 248), label, font=font(24, bold=True), fill=INK)
        for row_index, model in enumerate(MODELS):
            y = 304 + row_index * 54
            draw.text((left + 26, y), model, font=font(18, bold=True), fill=MODEL_COLORS[model])
            draw.text((left + 115, y), f"{values[model]:.4f}", font=font(19), fill=INK)
        best = max(values, key=values.get)
        draw.text(
            (left + 24, 508), f"最高：{best}", font=font(19, bold=True), fill=MODEL_COLORS[best]
        )
    draw.text(
        (76, 586),
        "P2 在混合故障均值最高；B5 在 63-mask 与持续故障分类均值更高。",
        font=font(22, bold=True),
        fill=INK,
    )
    frames.append(image)

    image, draw = base_frame(
        6,
        "Demo 2.0：真实产物、失败关闭",
        "页面五个标签页分别解释预测、机制、时间响应、总体背景和证据合同",
    )
    tabs = ("同样本对比", "质量与门控", "持续故障时间轴", "聚合背景", "产物合同")
    for index, label in enumerate(tabs):
        x = 74 + index * 226
        draw.rounded_rectangle(
            (x, 240, x + 198, 320), radius=18, fill="#F3F6FC", outline=BLUE, width=2
        )
        draw.text((x + 20, 266), label, font=font(19, bold=True), fill=BLUE)
    card(draw, (74, 382, 406, 548), "核心矩阵", "60 / 60", GREEN, "B4 / B5 / P / P2")
    card(draw, (474, 382, 806, 548), "扩展矩阵", "60 / 60", GREEN, "63-mask / 混合 / 持续")
    card(draw, (874, 382, 1206, 548), "随机回退", "禁止", RED, "合同不匹配直接报错")
    frames.append(image)

    image, draw = base_frame(
        7, "最终结论与发布边界", "项目适合面试演示与私有评审；公开发布仍需项目所有者决定"
    )
    card(draw, (74, 226, 338, 390), "核心实验", "60 / 60", GREEN)
    card(draw, (366, 226, 630, 390), "扩展实验", "60 / 60", GREEN)
    card(draw, (658, 226, 922, 390), "自动测试", "78 项", GREEN)
    card(draw, (950, 226, 1206, 390), "V1 保护", "813 / 813", GREEN)
    draw.rounded_rectangle((74, 438, 1206, 570), radius=20, fill="#FFF4F2", outline="#F0AAA0")
    draw.text((102, 464), "公开发布前的两个阻塞项", font=font(22, bold=True), fill=RED)
    draw.text(
        (102, 510),
        "① 选择代码许可证    ② 在 CITATION.cff 填写真实作者信息",
        font=font(27, bold=True),
        fill=INK,
    )
    draw.text(
        (74, 598),
        "诚实结论：P2 的价值在诊断与拒绝闭环，不是“所有指标都第一”。",
        font=font(22, bold=True),
        fill=PURPLE,
    )
    frames.append(image)

    return frames


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "reports/v2/demo_v2_walkthrough.gif"
    frames = make_frames(root)
    durations = [5000, 7000, 8000, 7000, 7000, 6000, 5000]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    with Image.open(output) as result:
        if result.n_frames != 7:
            raise RuntimeError(f"动图帧数异常：{result.n_frames}")
    print(f"已生成：{output}")
    print(f"帧数：{len(frames)}；总时长：{sum(durations) / 1000:.0f} 秒")


if __name__ == "__main__":
    main()
