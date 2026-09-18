"""Restore selected release assets with path, hash, and overwrite checks."""
from pathlib import Path,PurePosixPath
import argparse,gzip,hashlib,io,json,tarfile,urllib.request

ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    with path.open('rb') as f:
        h=hashlib.sha256()
        while block:=f.read(4*1024*1024):h.update(block)
        return h.hexdigest()

class JoinedParts(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.current=None
    def readable(self):return True
    def readinto(self,buffer):
        while True:
            if self.current is None:
                p=next(self.paths,None)
                if p is None:return 0
                self.current=p.open('rb')
            n=self.current.readinto(buffer)
            if n:return n
            self.current.close();self.current=None
    def close(self):
        if self.current:self.current.close()
        super().close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',action='append',required=True,help='Package ID; repeat for multiple packages.')
    p.add_argument('--from-dir',type=Path,help='Directory of downloaded release parts; otherwise download verified published URLs.')
    p.add_argument('--destination',type=Path,default=ROOT)
    p.add_argument('--cache',type=Path,default=ROOT/'.artifact-cache')
    a=p.parse_args();destination=a.destination.resolve()
    packages={x['id']:x for x in json.loads((ROOT/'metadata/artifact_packages.json').read_text())['packages']}
    expected=json.loads((ROOT/'metadata/artifact_files.json').read_text())['files']
    for package_id in a.package:
        if package_id not in packages:p.error('Unknown package: '+package_id)
        package=packages[package_id];paths=[]
        for part in package['parts']:
            path=(a.from_dir or a.cache)/part['name']
            if not path.is_file():
                if a.from_dir or not part['url']:p.error(f'{part["name"]} is unavailable. This release may not have been uploaded yet.')
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix(path.suffix+'.download')
                with urllib.request.urlopen(part['url']) as response,temporary.open('xb') as stream:
                    while block:=response.read(4*1024*1024):stream.write(block)
                if temporary.stat().st_size!=part['bytes'] or sha256(temporary)!=part['sha256']:raise ValueError('Downloaded part checksum mismatch: '+part['name'])
                temporary.rename(path)
            if path.stat().st_size!=part['bytes'] or sha256(path)!=part['sha256']:raise ValueError('Part checksum mismatch: '+part['name'])
            paths.append(path)
        count=0
        with io.BufferedReader(JoinedParts(paths)) as joined,gzip.GzipFile(fileobj=joined,mode='rb') as uncompressed,tarfile.open(fileobj=uncompressed,mode='r|') as archive:
            for member in archive:
                rel=PurePosixPath(member.name)
                if not member.isfile() or rel.is_absolute() or '..' in rel.parts or member.name not in expected:raise ValueError('Unexpected archive member: '+member.name)
                target=destination.joinpath(*rel.parts).resolve()
                if not target.is_relative_to(destination):raise ValueError('Archive path escapes the destination.')
                record=expected[member.name]
                if member.size!=record['bytes']:raise ValueError('Member size mismatch: '+member.name)
                count+=1
                if target.exists():
                    if target.is_file() and target.stat().st_size==record['bytes'] and sha256(target)==record['sha256']:continue
                    raise FileExistsError('Refusing to overwrite different content: '+str(target))
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_name(target.name+'.restoring')
                h=hashlib.sha256()
                with archive.extractfile(member) as source,temporary.open('xb') as out:
                    while block:=source.read(4*1024*1024):out.write(block);h.update(block)
                if h.hexdigest()!=record['sha256']:raise ValueError('File checksum mismatch: '+member.name)
                temporary.rename(target)
                target.chmod(member.mode & 0o777)
        if count!=package['files']:raise ValueError('Package file count mismatch.')
        print(f'Restored and verified {package_id}: {count} files')

if __name__=='__main__':main()
