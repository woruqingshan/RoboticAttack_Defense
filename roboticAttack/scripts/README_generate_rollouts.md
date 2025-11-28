# 批量生成多位置补丁视频脚本使用说明

## 功能

为22个补丁位置，每个位置运行 libero_object 的10个任务，每个任务执行1次，生成视频并保存到指定目录。

## 使用方法

### 基本用法

```bash
cd /root/autodl-tmp/code/roboticAttack

python scripts/generate_multi_position_rollouts.py \
    --patch-path /path/to/patch.pt \
    --cudaid 0
```

### 参数说明

- `--patch-path`: 补丁文件路径（必需）
- `--cudaid`: CUDA设备ID（默认：0）
- `--num-trials`: 每个任务的试验次数（默认：1）
- `--model`: 模型checkpoint路径（默认：openvla/openvla-7b-finetuned-libero-object）
- `--task-suite`: 任务套件名称（默认：libero_object）
- `--start-from`: 从指定位置索引开始（用于断点续传）

### 断点续传

如果某个位置失败，可以从失败的位置继续：

```bash
python scripts/generate_multi_position_rollouts.py \
    --patch-path /path/to/patch.pt \
    --cudaid 0 \
    --start-from 5  # 从第6个位置开始（索引从0开始）
```

## 输出结构

视频保存到：
```
rollouts/rollouts/libero_object/object_xy_{x}_{y}/{date}/*.mp4
```

例如：
- `rollouts/rollouts/libero_object/object_xy_0_0/2025_11_16/*.mp4`
- `rollouts/rollouts/libero_object/object_xy_10_10/2025_11_16/*.mp4`
- ...

## 进度记录

脚本会实时显示：
- 当前处理的位置索引和坐标
- 每个位置的执行命令
- 成功/失败统计
- 最终总结报告

## 22个补丁位置

脚本内置了22个位置：
- 左上区域：6个
- 右上区域：9个
- 左下区域：6个
- 右下区域：1个

详见脚本中的 `PATCH_POSITIONS` 列表。



