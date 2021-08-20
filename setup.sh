ROOT=$(dirname $(realpath ${0}))
mkdir ${ROOT}/enwiki ${ROOT}/glove ${ROOT}/bert ${ROOT}/data ${ROOT}/model ${ROOT}/other

wget -P ${ROOT} https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
wget -P ${ROOT} http://nlp.stanford.edu/data/glove.840B.300d.zip
wget -P ${ROOT} https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
wget -P ${ROOT} http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget -P ${ROOT} http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget -P ${ROOT} https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py

tar -xjvf ${ROOT}/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 --strip-components=1 -C ${ROOT}/enwiki
unzip -j ${ROOT}/glove.840B.300d.zip -d ${ROOT}/glove
unzip -j ${ROOT}/wwm_uncased_L-24_H-1024_A-16.zip -d ${ROOT}/bert
cp ${ROOT}/hotpot_train_v1.1.json ${ROOT}/data/train_dataset
cp ${ROOT}/hotpot_dev_distractor_v1.json ${ROOT}/data/develop_dataset
cp ${ROOT}/hotpot_evaluate_v1.py ${ROOT}/data/evaluate_script

rm ${ROOT}/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
rm ${ROOT}/glove.840B.300d.zip
rm ${ROOT}/wwm_uncased_L-24_H-1024_A-16.zip
rm ${ROOT}/hotpot_train_v1.1.json
rm ${ROOT}/hotpot_dev_distractor_v1.json
rm ${ROOT}/hotpot_evaluate_v1.py
