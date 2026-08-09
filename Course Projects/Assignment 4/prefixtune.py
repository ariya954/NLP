import argparse, os, string, sys, logging, re
import torch
import sacrebleu
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import PrefixTuningConfig, TaskType, get_peft_model, PeftModel
# import peft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TableToText:

    def __init__(
            self,
            modelfile,
            modelsuffix='.pt',
            basemodel='distilgpt2',
            traindata='e2e_nlg_cleaned',
            epochs=5,
            batchsize=4,
            lr=5e-5,
            virtualtokens=5,
            prefixprojection=False
        ):
        # the input sentences will be handled using this object, you do not need to manually encode input sentence words
        self.tokenizer = AutoTokenizer.from_pretrained(basemodel)
        self.tokenizer_pad_token_id = self.tokenizer.eos_token_id \
            if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        self.traindata = traindata
        self.modelfile = modelfile
        self.modelsuffix = modelsuffix
        self.basemodel = basemodel
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr
        self.virtualtokens = virtualtokens
        self.prefixprojection = prefixprojection
        self.prompt = "Convert the following table into English text: "
        self.training_data = []
        self.model = None # setup the model in self.decode() or self.train()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess_function(self, examples):
        text_column = "meaning_representation"
        label_column = "human_reference"
        max_length = 150
        batch_size = len(examples[text_column])
        inputs = [f"{self.prompt}{x} {self.tokenizer.bos_token} " for x in examples[text_column]]
        targets = [f"{x} {self.tokenizer.eos_token}" for x in examples[label_column]]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer_pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer_pad_token_id] * (
                    max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = \
                [0] * \
                (max_length - len(sample_input_ids)) + \
                model_inputs["attention_mask"][i]
            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def get_data(self, splits=("train", )):
        """
        Loads the requested dataset with name == :param dataset_name: and returns dataloaders over each split defined
          in :param splits: which can contain any subset of ("train", "validation", "test"). The dataloder batchsize will be
            defined using :param self.batchsize:.
        """
        dataset = load_dataset(self.traindata)
        processed_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset"
        )

        data_loaders = {}
        for split in splits:
            assert split in processed_datasets
            data_loaders[split] = DataLoader(
                                    processed_datasets[split],
                                    collate_fn=default_data_collator,
                                    batch_size=self.batchsize,
                                    pin_memory=True,
                                    shuffle=(split == "train")
                                  )
        return data_loaders

    def _clean_text(self, text):
        text = text.replace("\n", " ").replace("\r", " ")
        text = text.encode("ascii", errors="ignore").decode("ascii")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _parse_table(self, src):
        pairs = []
        for part in src.split("|"):
            part = part.strip()
            if ":" in part:
                k, v = part.split(":", 1)
                pairs.append((k.strip().lower(), v.strip().lower()))
        return pairs

    def _value_coverage_score(self, src, text):
        pairs = self._parse_table(src)
        text_l = text.lower()
        score = 0.0
        for key, value in pairs:
            if value and value in text_l:
                score += 2.0
            else:
                # small partial credit for multiword values
                pieces = [p for p in value.split() if p not in {"the", "a", "an", "-", "&"}]
                hit_count = sum(1 for p in pieces if p in text_l)
                if len(pieces) > 0:
                    score += 0.4 * (hit_count / len(pieces))
        return score

    def _repetition_penalty(self, text):
        toks = text.lower().split()
        if len(toks) <= 1:
            return 0.0
        unigram_rep = max(0, len(toks) - len(set(toks)))
        bigrams = list(zip(toks, toks[1:]))
        bigram_rep = max(0, len(bigrams) - len(set(bigrams)))
        return 0.08 * unigram_rep + 0.15 * bigram_rep

    def _length_score(self, text):
        n = len(text.split())
        if n < 6:
            return -2.0
        if n > 35:
            return -1.0
        return 0.0

    def _choose_best_candidate(self, src, candidates):
        best_text = ""
        best_score = -1e9
        for cand in candidates:
            cand = self._clean_text(cand)
            if not cand:
                continue
            score = 0.0
            score += self._value_coverage_score(src, cand)
            score += self._length_score(cand)
            score -= self._repetition_penalty(cand)
            if cand.endswith(",") or cand.endswith("|") or cand.endswith(":"):
                score -= 0.5
            if score > best_score:
                best_score = score
                best_text = cand
        return best_text

    def _fallback_from_table(self, src):
        pairs = dict(self._parse_table(src))
        name = pairs.get("name", "This restaurant")
        parts = []

        if "eat type" in pairs:
            parts.append(f"is a {pairs['eat type']}")
        if "food" in pairs:
            parts.append(f"serving {pairs['food']} food")
        if "area" in pairs:
            parts.append(f"in the {pairs['area']}")
        if "price range" in pairs:
            parts.append(f"with a {pairs['price range']} price range")
        if "customer rating" in pairs:
            parts.append(f"and has a {pairs['customer rating']} customer rating")
        if "family friendly" in pairs:
            ff = pairs["family friendly"]
            if ff == "yes":
                parts.append("and is family friendly")
            elif ff == "no":
                parts.append("and is not family friendly")
        if "near" in pairs:
            parts.append(f"near {pairs['near']}")

        if parts:
            text = f"{name} " + " ".join(parts) + "."
        else:
            text = src.replace(":", "").replace("|", "").replace("  ", " ")
        return self._clean_text(text)

    def _load_lines(self, inputfile):
        inputpath = Path(inputfile)
        assert inputpath.exists()
        with inputpath.open(encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if len(line) > 0 and not line.isspace()]
        return lines

    def _evaluate_bleu(self, model, inputfile, reffile):
        references = {}
        with open(reffile, 'r', encoding='utf-8') as ref:
            ref_data = [str(x) for x in ref.read().splitlines() if str(x)]
            for line in ref_data:
                src_id, _, suggested_reference = line.split('||')
                references.setdefault(src_id, [])
                references[src_id].append(suggested_reference)

        src_lines = self._load_lines(inputfile)
        output_data = []

        for i, src in enumerate(tqdm(src_lines, desc="eval decode")):
            hyp = self.predict(model, src, num_sequences=4)
            hyp = self._clean_text(hyp)
            if not hyp:
                hyp = self._fallback_from_table(src)
            output_data.append([str(i), hyp])

        score = 0.0
        total = 0.0
        bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)

        for line in output_data:
            r = references[line[0]]
            h = line[1]
            score += bleu_metric.sentence_score(h, r).score
            total += 1.0

        return score / total if total > 0 else 0.0

    def train(self):
        data_loaders = self.get_data(splits=("train", ))
        model = AutoModelForCausalLM.from_pretrained(self.basemodel)
        model.config.pad_token_id = self.tokenizer_pad_token_id

        # You can print the parameters for debugging or understanding the code
        # but make sure you comment it out otherwise it will pollute the output
        # that is produced for dev and test
        #model.print_trainable_parameters()

        # TODO
        # if using HF peft module, then add calls to PrefixTuningConfig and get_peft_model
        # which will take num_virtual_tokens which is set to self.virtualtokens and
        # prefix_projection which is set to self.prefixprojection
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.virtualtokens,
            prefix_projection=self.prefixprojection,
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(data_loaders["train"]) * self.epochs),
        )
        model = model.to(device)

        best_bleu = -1.0
        dev_input = os.path.join('data', 'input', 'dev.txt')
        dev_ref = os.path.join('data', 'reference', 'dev.out')

        for epoch in range(self.epochs):
            model.train()

            # TODO rest of the training steps for prefix tuning
            total_loss = 0.0
            progress_bar = tqdm(data_loaders["train"], desc=f"train epoch {epoch+1}/{self.epochs}")
            optimizer.zero_grad()

            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / max(1, progress_bar.n):.4f}")

            if epoch == self.epochs - 1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + epoch_str + self.modelsuffix
            model.save_pretrained(savefile)

            try:
                model.eval()
                bleu = self._evaluate_bleu(model, dev_input, dev_ref)
                print(f"epoch {epoch+1} dev.txt BLEU = {bleu:.4f}", file=sys.stderr)
                if bleu > best_bleu:
                    best_bleu = bleu
                    bestfile = self.modelfile + self.modelsuffix
                    model.save_pretrained(bestfile)
                    print(f"saved best checkpoint to {bestfile}", file=sys.stderr)
            except Exception as e:
                print(f"warning: could not evaluate dev.txt after epoch {epoch+1}: {e}", file=sys.stderr)

    def decode(self, model, inputfile):
        inputpath = Path(inputfile)
        assert inputpath.exists()
        with inputpath.open(encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if len(line) > 0 and not line.isspace()]
            decoder_output = []
            for i, src in tqdm(enumerate(lines), total=len(lines)):
                predicted_line = self.predict(model, src, num_sequences=4)
                #if not predicted_line or src.split()[0] not in predicted_line.split():
                    # if output generation failed then use a heuristic to generate some output
                    #predicted_line = src.replace(':', '').replace('|', '').replace('  ', ' ')
                predicted_line = self._clean_text(predicted_line)
                if not predicted_line:
                    predicted_line = self._fallback_from_table(src)
                decoder_output.append(f"{i}||{predicted_line}")
        return decoder_output

    def predict(self, model, src, num_sequences=1):
        inputs = self.tokenizer(self.prompt + src + ' ' + self.tokenizer.bos_token + ' ', return_tensors="pt")
        prediction = None
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=60,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer_pad_token_id,
                do_sample=False,
                num_beams=max(4, num_sequences),
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
                num_return_sequences=num_sequences
            )
            # TODO you may want to generate more than one sequence and choose the best one!
            texts = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

            cleaned = []
            for text in texts:
                text = text.lstrip().replace(self.prompt + src, "").replace("\n", " ")
                text = self._clean_text(text)
                if text:
                    cleaned.append(text)

            best = self._choose_best_candidate(src, cleaned)
            if not best:
                best = self._fallback_from_table(src)
            return best

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", dest="inputfile",
                            default=os.path.join('data', 'input', 'dev.txt'),
                             help="produce table to text output for these input tables")
    argparser.add_argument("-t", "--traindata", dest="traindata",
                            default='e2e_nlg_cleaned',
                            help="name of hugging face cleaned up dataset for the E2E table to text task")
    argparser.add_argument("-v", "--virtualtokens", dest="virtualtokens",
                            type=int, default=10,
                            help="number of virtual prompt tokens for prefix tuning")
    argparser.add_argument("-p", "--prefixprojection", dest="prefixprojection",
                            action="store_true", default=True,
                            help="whether to project the prefix embeddings")
    argparser.add_argument("-m", "--modelfile", dest="modelfile",
                            default=os.path.join('data', 'peft'),
                            help="filename without suffix for model files")
    argparser.add_argument("-s", "--modelsuffix", dest="modelsuffix", default='.pt',
                            help="filename suffix for model files")
    argparser.add_argument("-M", "--basemodel", dest="basemodel",
                            default='distilgpt2',
                            help="The base huggingface pretrained model to be used as the encoder.")
    argparser.add_argument("-e", "--epochs", dest="epochs", type=int, default=3,
                            help="number of epochs [default: 1]")
    argparser.add_argument("-b", "--batchsize", dest="batchsize", type=int, default=8,
                            help="batch size [default: 16]")
    argparser.add_argument("-r", "--lr", dest="lr", type=float, default=5e-5,
                            help="the learning rate used to finetune the BERT-like encoder module.")
    argparser.add_argument("-f", "--force", dest="force", action="store_true", default=False,
                            help="force training phase (warning: can be slow)")
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None,
                            help="log file for debugging")
    opts = argparser.parse_args()
    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)
    modelfile = opts.modelfile
    if modelfile.endswith('.pt'):
        modelfile = modelfile.removesuffix('.pt')
    table_to_text = TableToText(
                        modelfile,
                        modelsuffix=opts.modelsuffix,
                        basemodel=opts.basemodel,
                        traindata=opts.traindata,
                        epochs=opts.epochs,
                        batchsize=opts.batchsize,
                        lr=opts.lr,
                        virtualtokens=opts.virtualtokens,
                        prefixprojection=opts.prefixprojection
                    )
    # TODO default.py always uses a prompt to produce output from the pretrained model
    # when you have implemented prefix tuning then change this to False to train and/or 
    # use your prefix tuned model
    model = None
    if False:
        print(f"Loading the non-finetuned pre-trained model: {opts.basemodel}", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(opts.basemodel)
        model.config.pad_token_id = table_to_text.tokenizer_pad_token_id
        model = model.to(device)
    else:
        if not os.path.isdir(modelfile + opts.modelsuffix) or opts.force:
            print(f"Could not find modelfile {modelfile + opts.modelsuffix} or -f used. Starting training.", file=sys.stderr)
            table_to_text.train()
            print("Training done.", file=sys.stderr)
        # use the model file if available and opts.force is False
        load_dir = modelfile + opts.modelsuffix
        if not os.path.isdir(load_dir):
            for e in range(opts.epochs - 1, -1, -1):
                candidate = modelfile + str(e) + opts.modelsuffix
                if os.path.isdir(candidate):
                    load_dir = candidate
                    break
        assert(os.path.isdir(load_dir))
        print(f"Found modelfile {load_dir}. Starting decoding.", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(opts.basemodel)
        model.config.pad_token_id = table_to_text.tokenizer_pad_token_id
        # TODO: if using hf peft library for prefix tuning:
        # model = PeftModel.from_pretrained(model, modelfile + opts.modelsuffix)
        model = PeftModel.from_pretrained(model, load_dir)
        model = model.to(device)
    if model:
        decoder_output = table_to_text.decode(model, opts.inputfile)
        print("\n".join(decoder_output))