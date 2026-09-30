import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.publish_documents import publish_reports


def response(data, status=200):
    result = MagicMock()
    result.status_code = status
    result.json.return_value = data
    return result


class TestPublishDocuments(unittest.TestCase):
    def test_creates_category_and_uploads_both_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_report.html").write_text("<html></html>")
            (root / "model_report.xlsx").write_bytes(b"excel")
            session = MagicMock()
            session.__enter__.return_value = session
            session.post.side_effect = [
                response({"accessToken": "token"}),
                response({"id": "topic-1", "name": "Pruebas rapidas lis"}, 201),
                response([{"success": True}, {"success": True}]),
            ]
            session.get.return_value = response([])
            with patch.dict(os.environ, {"DOCUMENT_API_CLIENT_ID": "id",
                                      "DOCUMENT_API_CLIENT_SECRET": "secret"}), \
                 patch("tools.publish_documents.requests.Session", return_value=session):
                publish_reports(root, "lis_yolox", "1.0.0")
            self.assertEqual(session.post.call_args_list[1].kwargs["json"]["name"], "Pruebas rapidas lis")
            upload = session.post.call_args_list[2]
            self.assertEqual(upload.kwargs["data"], {"topicId": "topic-1"})
            self.assertEqual([part[1][0] for part in upload.kwargs["files"]], [
                "lis_yolox_1.0.0_model_report.html", "lis_yolox_1.0.0_model_report.xlsx"])

    def test_existing_category_and_partial_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_report.html").write_text("html")
            (root / "model_report.xlsx").write_bytes(b"excel")
            session = MagicMock()
            session.__enter__.return_value = session
            session.post.side_effect = [response({"accessToken": "token"}),
                                        response([{"success": True}, {"success": False}])]
            session.get.return_value = response([{"id": "vet-1", "name": "veterinaria"}])
            with patch.dict(os.environ, {"DOCUMENT_API_CLIENT_ID": "id",
                                      "DOCUMENT_API_CLIENT_SECRET": "secret"}), \
                 patch("tools.publish_documents.requests.Session", return_value=session):
                with self.assertRaisesRegex(RuntimeError, "no confirmó ambos"):
                    publish_reports(root, "vet_yolox", "1.0.0")
            self.assertEqual(session.post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
