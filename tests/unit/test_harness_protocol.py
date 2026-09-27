"""Guard the upstream task fixes required by our evaluation protocol."""

from types import SimpleNamespace

import pytest

pytest.importorskip("lm_eval")
from lm_eval.tasks.squad_completion.task import SQUADCompletion


def test_squad_completion_strips_prompt_and_scored_target():
    # No dataset download is needed to exercise the actual upstream methods.
    task = object.__new__(SQUADCompletion)
    doc = {"text": " " * 1800 + "Passage.The answer is  ", "value": " answer \n"}
    assert task.doc_to_text(doc) == "Passage.The answer is"
    assert task.doc_to_target(doc) == "answer"
    assert task.process_results(doc, ["answer"]) == {"contains": 1}
    assert task.process_results(doc, ["wrong"]) == {"contains": 0}
    assert task.VERSION == 1


def test_squad_task_iterator_partitions_documents_with_local_indices():
    # This only checks task request construction. The evaluator maps logged
    # sample doc_id values back to global indices after scoring.
    task = object.__new__(SQUADCompletion)
    task._config = SimpleNamespace(task="squad_completion")
    task.dataset = {"validation": [{"doc_id": str(i)} for i in range(8)]}
    shards = [list(task.doc_iterator(samples=list(range(start, start + 4)))) for start in (0, 4)]
    assert [[i for i, _ in shard] for shard in shards] == [list(range(4)), list(range(4))]
    assert [doc for shard in shards for _, doc in shard] == task.dataset["validation"]
