import json
import unittest

import torch

from common import ROOT, config, load_tokenizer, read_jsonl
from evaluate import grade
from sft_data import SFTDataset, SFTCollator, answer_loss_sum, prompt_ids


class SFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer()
        cls.rows = read_jsonl(ROOT / "data" / "train.jsonl")
        cls.ds = SFTDataset(cls.rows, cls.tokenizer, config()["max_length"])

    def test_mask_and_eos(self):
        eos = self.tokenizer.eos_token_id
        for row, e in zip(self.rows, self.ds.examples):
            p = prompt_ids(self.tokenizer, row)
            self.assertEqual(e["input_ids"][:len(p)], p)
            self.assertTrue(all(y == -100 for y in e["labels"][:len(p)]))
            self.assertEqual(e["labels"][len(p):], e["input_ids"][len(p):])
            self.assertEqual(e["labels"][-1], eos)
            self.assertGreater(len(e["labels"]) - len(p), 1)

    def test_padding_mask_is_separate_from_attention(self):
        batch = SFTCollator(self.tokenizer.pad_token_id)([self.ds[0], self.ds[1]])
        for i,e in enumerate([self.ds[0],self.ds[1]]):
            n, p = len(e["input_ids"]), e["prompt_length"]
            self.assertTrue(bool(batch["attention_mask"][i,:p].eq(1).all()))
            self.assertTrue(bool(batch["labels"][i,:p].eq(-100).all()))
            self.assertTrue(bool(batch["attention_mask"][i,n:].eq(0).all()))
            self.assertTrue(bool(batch["labels"][i,n:].eq(-100).all()))

    def test_shift_and_loss_mask(self):
        logits = torch.randn(1,5,7,requires_grad=True)
        labels = torch.tensor([[-100,-100,-100,3,4]])
        loss,n = answer_loss_sum(logits, labels)
        expected = torch.nn.functional.cross_entropy(logits[0,2:4], torch.tensor([3,4]), reduction="sum")
        self.assertEqual(n,2)
        self.assertTrue(torch.allclose(loss,expected))
        loss.backward()
        self.assertEqual(float(logits.grad[0,:2].abs().sum()),0.0)
        self.assertGreater(float(logits.grad[0,2:4].abs().sum()),0.0)

    def test_token_normalized_accumulation(self):
        torch.manual_seed(2)
        x1,x2 = torch.randn(3,2),torch.randn(5,2)
        y1,y2 = torch.tensor([0,1,0]),torch.tensor([1,1,0,0,1])
        w = torch.randn(2,2,requires_grad=True)
        total = (torch.nn.functional.cross_entropy(x1@w,y1,reduction="sum")+torch.nn.functional.cross_entropy(x2@w,y2,reduction="sum"))/8
        expected = torch.autograd.grad(total,w)[0]
        w2=w.detach().clone().requires_grad_()
        (torch.nn.functional.cross_entropy(x1@w2,y1,reduction="sum")/8).backward()
        (torch.nn.functional.cross_entropy(x2@w2,y2,reduction="sum")/8).backward()
        self.assertTrue(torch.allclose(expected,w2.grad,atol=1e-6))

    def test_no_overlap_in_prompt_template_or_order(self):
        sets=[];templates=[];orders=[]
        for split in ("train","val","test"):
            rows=read_jsonl(ROOT/"data"/f"{split}.jsonl")
            sets.append({r["messages"][1]["content"] for r in rows})
            templates.append({r["meta"]["template"] for r in rows})
            orders.append({r["target"]["order_id"] for r in rows if r["target"]["order_id"] is not None})
        for a,b in ((0,1),(0,2),(1,2)):
            self.assertFalse(sets[a]&sets[b]);self.assertFalse(templates[a]&templates[b]);self.assertFalse(orders[a]&orders[b])

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):
            SFTDataset(self.rows[:1],self.tokenizer,4)

    def test_eos_not_masked_when_pad_has_same_id(self):
        eos=self.tokenizer.eos_token_id
        batch=SFTCollator(eos)([self.ds[0],self.ds[1]])
        for i,e in enumerate([self.ds[0],self.ds[1]]):
            self.assertEqual(int(batch["labels"][i,len(e["input_ids"])-1]),eos)
            self.assertTrue(bool(batch["labels"][i,len(e["input_ids"]):].eq(-100).all()))

    def test_strict_evaluator(self):
        target=self.rows[0]["target"]
        text=json.dumps(target,ensure_ascii=False)
        self.assertTrue(grade(text,target)["exact_match"])
        self.assertFalse(grade('```json\n'+text+'\n```',target)["schema_valid"])
        bad={**target,"urgent":int(target["urgent"])}
        self.assertFalse(grade(json.dumps(bad),target)["exact_match"])
        self.assertFalse(grade(json.dumps({**target,"intent":[]}),target)["schema_valid"])
        self.assertFalse(grade('{"urgent":true,"urgent":false}',target)["json_valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
