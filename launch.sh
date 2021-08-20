ROOT=$(dirname $(realpath ${0}))

if [ ${#} -eq 0 ] || [ ${1} -eq 0 ]; then
  python ${ROOT}/preprocess.py 0
  python ${ROOT}/preprocess.py 1
  python ${ROOT}/preprocess.py 2
  python ${ROOT}/preprocess.py 3
fi

if [ ${#} -eq 0 ] || [ ${1} -eq 1 ]; then
  python ${ROOT}/construct.py
  python ${ROOT}/optimize.py
fi

if [ ${#} -eq 0 ] || [ ${1} -eq 2 ]; then
  python ${ROOT}/execute.py ${ROOT}/data/develop_dataset ${ROOT}/data/develop_solution
  python ${ROOT}/data/evaluate_script ${ROOT}/data/develop_solution ${ROOT}/data/develop_dataset
fi
