# 4 grams
mkdir gram
for i in {1..9}
do
    python unique_gram.py $i
done
python join_grams.py

# byte count
python freq_count.py

# instruction count
python instr_freq.py

# dll features
python dll.py

## all commands above can run in pypy instead, and it can be much faster.
## just change all python into pypy

# asm image features (cannot run it in pypy)
python image_fea.py


######
#generate your 4k features here maybe
python get_id.py
python gen_opcount_seg.py
######



## model
python semi_model.py
