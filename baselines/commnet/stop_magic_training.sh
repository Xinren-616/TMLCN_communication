#!/bin/bash

# 停止所有MAGIC训练进程

echo "正在查找MAGIC训练进程..."
PIDS=$(ps aux | grep "train_with_online_shapley.*magic" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "没有找到正在运行的MAGIC训练进程"
    exit 0
fi

echo "找到以下MAGIC训练进程:"
echo "$PIDS"
echo ""
echo "正在停止这些进程..."

for PID in $PIDS; do
    echo "停止进程 $PID..."
    kill $PID
done

echo ""
echo "等待进程结束..."
sleep 3

# 检查是否还有残留进程
REMAINING=$(ps aux | grep "train_with_online_shapley.*magic" | grep -v grep | awk '{print $2}')
if [ ! -z "$REMAINING" ]; then
    echo "警告: 仍有进程未结束，使用kill -9强制终止..."
    for PID in $REMAINING; do
        echo "强制终止进程 $PID..."
        kill -9 $PID
    done
fi

echo ""
echo "所有MAGIC训练进程已停止"




