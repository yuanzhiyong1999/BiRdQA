#==========zh=======================
#python start.py \
#  --model mine.bert-base-zh-56 \
#  --dataset BiRdQA

#python start.py \
#  --model bert-wwm-ext \
#  --dataset BiRdQA

#python start.py \
#  --model roberta-wwm-ext-large \
#  --dataset BiRdQA

#python start.py \
#  --model ernie \
#  --dataset BiRdQA

#=============en========================
python start.py \
  --model other.albert-xxl \
  --dataset BiRdQA

#python start.py \
#  --model bert-large-en \
#  --dataset BiRdQA

#python start.py \
#  --model roberta-large \
#  --dataset BiRdQA

#python start.py \
#  --model albert-xxl \
#  --dataset BiRdQA

#python start.py \
#  --model unifiedqa-t5-large \
#  --dataset BiRdQA


#  使用read命令达到类似bat中的pause命令效果
echo 按任意键继续
read -n 1
echo 继续运行