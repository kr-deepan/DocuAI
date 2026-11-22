"""
Relational Database Manager (SQLite)
Provides persistent storage for isolated multi-project workspaces and document metadata.
"""
import os
import sqlite3
import datetime
from typing import List, Dict, Any, Optional

from src.config import PROJECTS_DB_PATH

class DatabaseManager:
    def __init__(self, db_path: str = PROJECTS_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        """Initializes tables for projects and documents."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER DEFAULT 0,
                    content TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)
            conn.commit()
        
        # Auto-seed the "default" project for existing legal corpus if not present
        self._seed_default_project()

    def _seed_default_project(self):
        """Ensures the default project exists and populates sample document metadata if empty."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT id FROM projects WHERE id = 'default'").fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO projects (id, name, description) VALUES (?, ?, ?)",
                    ("default", "Default Legal Corpus", "Pre-indexed enterprise contract and SLA sample documents")
                )
                conn.commit()

            doc_count = conn.execute("SELECT COUNT(*) FROM documents WHERE project_id = 'default'").fetchone()[0]
            if doc_count == 0:
                sample_dir = os.path.join(os.path.dirname(self.db_path), "sample_docs")
                if not os.path.exists(sample_dir):
                    sample_dir = os.path.join("data", "sample_docs")
                if os.path.exists(sample_dir):
                    import glob
                    files = glob.glob(os.path.join(sample_dir, "*.*"))
                    for f in files:
                        fname = os.path.basename(f)
                        ext = os.path.splitext(fname)[1].lstrip(".")
                        size = os.path.getsize(f)
                        try:
                            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                                content = fp.read()
                        except Exception:
                            content = f"[Document content from {fname}]"
                        conn.execute(
                            "INSERT OR IGNORE INTO documents (id, project_id, filename, file_type, file_size, content, chunk_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (f"default_{fname}", "default", fname, ext, size, content, 3)
                        )
                    conn.commit()

    # -------------------------------------------------------------------------
    # Project CRUD Operations
    # -------------------------------------------------------------------------

    def create_project(self, project_id: str, name: str, description: str = "") -> Dict[str, Any]:
        """Creates a new isolated project workspace."""
        clean_id = project_id.strip().lower().replace(" ", "_")
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, description) VALUES (?, ?, ?)",
                (clean_id, name.strip(), description.strip())
            )
            conn.commit()
        return self.get_project(clean_id)

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single project record by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            if row:
                return dict(row)
        return None

    def list_projects(self) -> List[Dict[str, Any]]:
        """Returns all projects with their document and chunk counts."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT 
                    p.id, 
                    p.name, 
                    p.description, 
                    p.created_at,
                    COUNT(d.id) AS document_count,
                    COALESCE(SUM(d.chunk_count), 0) AS total_chunks
                FROM projects p
                LEFT JOIN documents d ON p.id = d.project_id
                GROUP BY p.id
                ORDER BY CASE WHEN p.id = 'default' THEN 0 ELSE 1 END, p.created_at DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def delete_project(self, project_id: str) -> bool:
        """Deletes a project and cascades deletion to its documents."""
        if project_id == "default":
            # Protect default project from deletion
            return False
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Document CRUD Operations
    # -------------------------------------------------------------------------

    def save_document(
        self,
        doc_id: str,
        project_id: str,
        filename: str,
        content: str,
        file_type: str = "txt",
        file_size: int = 0,
        chunk_count: int = 0
    ) -> Dict[str, Any]:
        """Saves an uploaded document and its extracted text to a specific project."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO documents 
                (id, project_id, filename, file_type, file_size, content, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (doc_id, project_id, filename, file_type, file_size, content, chunk_count))
            conn.commit()
        return self.get_document(doc_id)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a document record by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if row:
                return dict(row)
        return None

    def get_project_documents(self, project_id: str) -> List[Dict[str, Any]]:
        """Returns all documents belonging to a project."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, project_id, filename, file_type, file_size, chunk_count, created_at FROM documents WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document from the database."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            return cursor.rowcount > 0

# Singleton instance
db_manager = DatabaseManager()
