#python run.py \
#  --model bert-base-zh \
#  --dataset BiRdQA

#python run.py \
#  --model bert-base-en \
#  --dataset BiRdQA

python run.py \
  --model bert-wwm-ext \
  --dataset BiRdQA

#python run.py \
#  --model roberta-wwm-ext-large \
#  --dataset BiRdQA

#python run.py \
#  --model ernie \
#  --dataset BiRdQA


#  使用read命令达到类似bat中的pause命令效果
echo 按任意键继续
read -n 1
echo 继续运行