• 可以，已经支持了。

  现在总调度脚本 multi_train/script/schedule_composable_policy_followup_3nodes.sh 的行为是：

  - 默认分区列表：
      - dexmanip,ftmanip
  - 默认资源请求：
      - --gres=gpu:8
  - 三个 follow-up 作业会一起提交：
      - toxicity
      - ethics
      - truth + bbq

  这样 Slurm 会自己决定：

  - 哪个作业先落到 dexmanip
  - 哪个作业先落到 ftmanip
  - 如果两个分区都暂时没 8 卡空位，作业就挂起等待
  - 一旦任一分区有空位，就会自动占上

  也就是说，你要的“默认八张一开，如果有空卡就占上”，本质上就是：

  -p dexmanip,ftmanip --gres=gpu:8

  然后把多个作业一起提交，让调度器自己选最快有资源的分区。

  你现在可以直接这么用：

  bash multi_train/script/schedule_composable_policy_followup_3nodes.sh

  如果你想显式指定，也可以：

  PARTITIONS=dexmanip,ftmanip GRES=gpu:8 bash multi_train/script/schedule_composable_policy_followup_3nodes.sh

  如果以后想加别的 srun 参数，我也预留了：

  SRUN_EXTRA_ARGS="--cpus-per-task=8 --mem=0" bash multi_train/script/schedule_composable_policy_followup_3nodes.sh

  至于你前面问的 evaluate_composable_toxicity_policy_followup.sh 为什么四张卡后要 wait：

  - 那不是算法依赖
  - 只是当时我把 8 个实验分成两批 4 个，避免默认情况下同时起 8 个大模型把机器打满
  - 现在这个脚本我已经改成了更通用的版本：
      - 用 GPU_IDS 指定可用卡
      - 用 MAX_PARALLEL 控制同时跑多少个
  - 所以不再固定“4 张卡后 wait”，而是“达到 MAX_PARALLEL 后 wait”

  如果你愿意，我下一步可以再给这个总调度脚本加一个更高级的模式：

  - 先自动建 intervention stats
  - stats 成功后再自动发 follow-up 3 个作业
  - 变成一键全流程脚本。
