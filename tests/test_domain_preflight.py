import tempfile
import unittest
from pathlib import Path
from scripts.preflight_domain_objects import check_objects


class ObjectPreflightTests(unittest.TestCase):
    def test_missing_vs_access_error_and_no_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'selection.csv'
            p.write_text('id,video path\na,videos/group/a.mp4\nb,videos/group/b.mp4\nc,videos/group/c.mp4\n')
            def exists(path):
                if path.endswith('c.mp4'):raise PermissionError('denied')
                return path.endswith('a.mp4')
            result=check_objects([p,p],'obs://bucket/videos',exists)
            self.assertEqual(result['checked'],3)
            self.assertEqual(result['failed'],2)
            self.assertEqual({r['reason'] for r in result['failures']},{'not_found','access_error'})
            self.assertFalse(result['media_downloaded'])
            self.assertTrue(check_objects([p],'root',lambda path:True)['passed'])


if __name__=='__main__':unittest.main()
