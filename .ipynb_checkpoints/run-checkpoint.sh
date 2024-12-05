for head in 1 4 8
do
for r in  'none' 'relu'
do
for batch in 64 128 256 
do
# 'PROTEINS' 'MUTAG' 'IMDB-BINARY' 
for d in 'IMDB-MULTI' 'IMDB-BINARY' 'NCI1' 'Mutagenicity' 'MUTAG' 'PROTEINS' 'DD' 
# for d in 'Mutagenicity'
# for d in 'NCI1'
do 
python main.py --dataset $d --read_op weight_sum --device 0 --drop 0.5 --relu $r --batch_size $batch --head $head
done
done
done
done