source .env

huggingface-cli login --token $DATA_E2E_HUGGINGFACE_TOKEN

huggingface-cli repo create $1 --type model -y

git lfs install

git clone https://${DATA_E2E_HUGGINGFACE_USERNAME}:${DATA_E2E_HUGGINGFACE_TOKEN}@huggingface.co/${DATA_E2E_HUGGINGFACE_USERNAME}/$1

cd $1

huggingface-cli lfs-enable-largefiles .

cp ../resources/huggingface/README.md .

git add README.md

git commit -m "Initial commit"

git push

cd -

rm -rf $1
