import datasets
from loguru import logger as lg
import pandas as pd

logger = datasets.logging.get_logger(__name__)

_SENTENCE = "sentence"
_SEN1 = "sent1"
_SEN2 = "sent2"
_NEG  = "hard_neg"
_SCORE = "score"

_NLI_PATH = "nli_for_simcse.csv"
_WIKI_PATH = "wiki1m_for_simcse.txt"


class MyDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name='nli',
            description='nli for simcse'
        ),
        datasets.BuilderConfig(
            name='wiki',
            description='wiki1m for simcse'
        ),
        datasets.BuilderConfig(
            name='sts',
            description='stsbenchmark for simcse'
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.name == 'nli':
            return datasets.DatasetInfo(
                description=self.config.description,
                features=datasets.Features({
                    _SEN1: datasets.Value("string"),
                    _SEN2: datasets.Value("string"),
                    _NEG: datasets.Value("string"),
                })
            )
        if self.config.name == 'wiki':
            return datasets.DatasetInfo(
                description=self.config.description,
                features=datasets.Features({
                    _SENTENCE: datasets.Value("string"),
                })
            )
        if self.config.name == 'sts':
            return datasets.DatasetInfo(
                description=self.config.description,
                features=datasets.Features({
                    _SEN1: datasets.Value("string"),
                    _SEN2: datasets.Value("string"),
                    _SCORE: datasets.Value("float"),
                })
            )
        
    def _split_generators(self, dl_manager: datasets.DownloadManager):
        if self.config.name == 'nli':
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        'file_path': _NLI_PATH,
                    }
                ),
            ]
        if self.config.name == 'wiki':
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        'file_path': _WIKI_PATH,
                    }
                ),
            ]
        if self.config.name == "sts":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        'file_path': "./SentEval/STS/STSBenchmark/sts-dev.csv",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        'file_path': "./SentEval/STS/STSBenchmark/sts-test.csv"
                    },
                ),
            ]
        
    def _generate_examples(self, file_path):
        if self.config.name == 'nli':
            df = pd.read_csv(file_path, sep=',')
            rows = df.to_dict('records')
            for idx, row in enumerate(rows):
                example = {
                    _SEN1: row['sent0'].strip(),
                    _SEN2: row['sent0'].strip(),
                    _NEG: row['hard_neg'].strip(),
                }
                yield idx, example

        if self.config.name == 'wiki':
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.readlines()
            for idx, line in enumerate(lines):
                example = {
                    _SENTENCE: line.strip()
                }
                yield idx, example
        if self.config.name == "sts":
            with open(file_path, 'r') as f:
                lines = f.readlines()
            for idx, line in enumerate(lines):
                row = line.strip().split('\t')
                example = {
                    _SEN1: row[5].strip(),
                    _SEN2: row[6].strip(),
                    _SCORE: float(row[4]),
                }
                yield idx, example



def load_mydataset(name):
    dataset = datasets.load_dataset(path="./load_datasets.py", name=name, trust_remote_code=True)
    lg.info("🍻 MyDataset's name: {}\n{}".format(name, dataset))
    dataset.save_to_disk(name)
    return

if __name__ == "__main__":

    # ###################### name='nli' #########################
    # DatasetDict({
    #     train: Dataset({
    #         features: ['sent1', 'sent2', 'hard_neg'],
    #         num_rows: 275601
    #     })
    # })
    # ###########################################################

    # ###################### name='wiki' ########################
    # DatasetDict({
    #     train: Dataset({
    #         features: ['sentence'],
    #         num_rows: 1000000
    #     })
    # })
    # ###########################################################

    # ###################### name='sts' ########################
    # DatasetDict({
    #     validation: Dataset({
    #         features: ['sent1', 'sent2', 'score'],
    #         num_rows: 1500
    #     })
    #     test: Dataset({
    #         features: ['sent1', 'sent2', 'score'],
    #         num_rows: 1379
    #     })
    # })
    # ###########################################################

    load_mydataset(name='nli')
    load_mydataset(name='wiki')
    load_mydataset(name='sts')
    lg.info("✅ All datasets have been loaded successfully!")
