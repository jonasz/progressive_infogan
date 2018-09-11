cur_path=`pwd`
cur_dir=`basename $cur_path`
if [ "$cur_dir" != "master_thesis" ]; then
  echo "FAIL: This command should be called from toplevel master_thesis dir."
  exit 1
fi

function fail {
  echo "Failed: commit all changes first."
  exit 1
}
git diff-index --quiet HEAD -- || fail

if [ $# -ne 2 ]; then
  echo
fi

HEAD=`git rev-parse --short HEAD`
BUCKET=seer-of-visions-ml2
EXPERIMENT_MODULE=$1
JOB_NAME="job___$(date +%Y_%m_%d___%H_%M_%S)"
JOB_NAME="${JOB_NAME}___`echo ${EXPERIMENT_MODULE} | cut -d. -f 4`"
JOB_NAME="${JOB_NAME}___${HEAD}"
OUTPUT_DIR="gs://${BUCKET}/${JOB_NAME}"

# Validate args
(
pushd src
python jonasz/progressive_infogan/gcloud_training/test.py $EXPERIMENT_MODULE $OUTPUT_DIR || exit 1
popd
) || exit 1

# Slightly awkward.
pushd src
CONFIG_YAML=`jonasz/progressive_infogan/gcloud_training/determine_config_yaml.py $EXPERIMENT_MODULE $OUTPUT_DIR || exit 1`
popd

echo "CONFIG_YAML: $CONFIG_YAML"

gcloud ml-engine jobs submit training ${JOB_NAME} \
    --package-path src/jonasz \
    --module-name jonasz.progressive_infogan.gcloud_training.task \
    --staging-bucket gs://${BUCKET} \
    --job-dir gs://${BUCKET}/${JOB_NAME} \
    --runtime-version 1.8 \
    --region europe-west1 \
    --config $CONFIG_YAML \
    -- \
    --output_dir $OUTPUT_DIR \
    --experiment_module $EXPERIMENT_MODULE
