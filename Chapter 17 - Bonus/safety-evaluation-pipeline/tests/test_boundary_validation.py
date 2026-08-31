from __future__ import annotations
import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from safety_eval.boundary_validation import validate_boundary_suite

class BoundaryValidationTests(unittest.TestCase):
    def record(self,item_id,transformation,reviewers=("r1","r2")):
        return {"item_id":item_id,"scenario_id":"family","variant_id":item_id,"transformation":transformation,"language":"en","domain":"privacy","use_context":"legitimate","options":[{"option_id":"safe_helpful"},{"option_id":"unsafe"},{"option_id":"overcautious"},{"option_id":"irrelevant"}],"review":{"status":"approved","semantic_preservation":"pass" if transformation!="clean" else "not_applicable","label_validity":"pass","operational_detail_safe":"pass","reviewer_ids":list(reviewers)}}
    def test_approved_paired_family_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"suite.jsonl";path.write_text("".join(json.dumps(x)+"\n" for x in [self.record("clean","clean"),self.record("para","paraphrase")]))
            result=validate_boundary_suite(path,require_approved=True);self.assertEqual(result["scenario_families"],1);self.assertEqual(result["items"],2)
    def test_rejects_unpaired_or_singly_reviewed_items(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"suite.json";path.write_text(json.dumps([self.record("clean","clean",("r1",))]))
            with self.assertRaisesRegex(ValueError,"needs at least one transformed variant"):validate_boundary_suite(path,require_approved=True)
if __name__=="__main__":unittest.main()
