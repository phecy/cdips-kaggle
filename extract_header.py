# coding : utf-8
import sys

def main(in_file='avito_train.tsv'):
    with open(in_file) as in_fid:
        head = in_fid.readline().strip().split('\t')
        col_idx = zip(range(len(head)),head)
        with open('header.txt','w') as out_fid:
            for item in col_idx:
                 out_fid.write('\t'.join(map(str,list(item)))+'\n')
   #df = pd.read_csv(in_file,'\t')

if __name__=="__main__":            
    if len(sys.argv)>1:
        main(sys.argv[1])
    else:
        main()
