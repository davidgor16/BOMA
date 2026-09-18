"""Verify publication-file hashes and the executable AST of annotated Python files."""
from pathlib import Path
import ast,hashlib,json

ROOT=Path(__file__).resolve().parents[1]

class WithoutDocstrings(ast.NodeTransformer):
    def visit_scope(self,node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
        return node
    visit_Module=visit_scope
    visit_ClassDef=visit_scope
    visit_FunctionDef=visit_scope
    visit_AsyncFunctionDef=visit_scope

def executable_hash(data):
    tree=WithoutDocstrings().visit(ast.parse(data))
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()

def main():
    manifest=json.loads((ROOT/'metadata/source_manifest.json').read_text())
    failures=[]
    for rel,expected in manifest['files'].items():
        path=ROOT/rel
        if not path.is_file():failures.append([rel,'missing']);continue
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected['publication_sha256']:failures.append([rel,'publication hash mismatch'])
        if 'executable_ast_sha256' in expected and executable_hash(data)!=expected['executable_ast_sha256']:failures.append([rel,'executable AST mismatch'])
    print(json.dumps({'checked_source_files':len(manifest['files']),'failures':failures},indent=2))
    raise SystemExit(bool(failures))

if __name__=='__main__':main()
