import 'server-only';
import { spawn, type ChildProcess } from 'child_process';
import { mkdir, readdir, readFile, stat, writeFile } from 'fs/promises';
import path from 'path';
import type { JobSummary, PipelineEvent } from '@/lib/types';

/*
 * Paths are assembled with Array.join (not path.join on literals) so the bundler's
 * file tracer doesn't try to follow the Python virtualenv or the outputs folder.
 */
const SEP = path.sep;

export function repoRoot(): string {
  return process.env.HEIMDALL_ROOT ?? path.resolve(process.cwd(), '..');
}

export function pythonExe(): string {
  if (process.env.HEIMDALL_PYTHON) return process.env.HEIMDALL_PYTHON;
  const venv = [repoRoot(), 'venv'];
  return process.platform === 'win32'
    ? [...venv, 'Scripts', 'python.exe'].join(SEP)
    : [...venv, 'bin', 'python'].join(SEP);
}

export function jobsDir(): string {
  return process.env.HEIMDALL_JOBS_DIR ?? [repoRoot(), 'outputs', 'jobs'].join(SEP);
}

const ID_RE = /^[a-z0-9-]{4,64}$/;
const FILE_RE = /^[A-Za-z0-9._-]{1,128}$/;

export function jobPath(id: string, file?: string): string {
  if (!ID_RE.test(id)) throw new Error('invalid job id');
  if (file !== undefined && (!FILE_RE.test(file) || file.startsWith('.'))) throw new Error('invalid file name');
  return file === undefined ? [jobsDir(), id].join(SEP) : [jobsDir(), id, file].join(SEP);
}

export function newJobId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export const JOB_NAME = 'result';

export function safeExt(name: string, allowed: string[], fallback: string): string {
  const ext = path.extname(name).toLowerCase();
  return allowed.includes(ext) ? ext : fallback;
}

export async function writeJobInfo(id: string, info: Partial<JobSummary> & Record<string, unknown>) {
  const file = jobPath(id, 'job.json');
  let prev: Record<string, unknown> = {};
  try {
    prev = JSON.parse(await readFile(file, 'utf8'));
  } catch {}
  await writeFile(file, JSON.stringify({ ...prev, ...info, id }, null, 2));
}

export async function listJobs(limit = 20): Promise<JobSummary[]> {
  let entries: string[] = [];
  try {
    entries = await readdir(jobsDir());
  } catch {
    return [];
  }
  const jobs: JobSummary[] = [];
  for (const id of entries) {
    if (!ID_RE.test(id)) continue;
    try {
      const info = JSON.parse(await readFile(jobPath(id, 'job.json'), 'utf8')) as JobSummary;
      jobs.push(info);
    } catch {}
  }
  return jobs.sort((a, b) => (a.created < b.created ? 1 : -1)).slice(0, limit);
}

export async function ensureDir(dir: string) {
  await mkdir(dir, { recursive: true });
}

export async function fileExists(p: string): Promise<boolean> {
  try {
    return (await stat(p)).isFile();
  } catch {
    return false;
  }
}

/**
 * Run a Heimdall Python script. Lines on stdout prefixed with "@@HEIMDALL " are parsed
 * as structured events; everything else (stdout and stderr — Python logging writes to
 * stderr) is forwarded as log lines.
 */
export function runPython(
  script: string,
  args: string[],
  handlers: { onEvent?: (e: PipelineEvent) => void; onLog?: (line: string, isErr: boolean) => void },
): { child: ChildProcess; done: Promise<number> } {
  const child = spawn(pythonExe(), [[repoRoot(), script].join(SEP), ...args], {
    cwd: repoRoot(),
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
  });

  const makeReader = (isErr: boolean) => {
    let buf = '';
    return (chunk: Buffer) => {
      buf += chunk.toString('utf8');
      let nl: number;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).replace(/\r$/, '');
        buf = buf.slice(nl + 1);
        if (!line.trim()) continue;
        if (line.startsWith('@@HEIMDALL ')) {
          try {
            handlers.onEvent?.(JSON.parse(line.slice(11)) as PipelineEvent);
            continue;
          } catch {}
        }
        // Drop ANSI colour codes some libraries emit.
        handlers.onLog?.(line.replace(/\x1b\[[0-9;]*m/g, ''), isErr);
      }
    };
  };
  child.stdout?.on('data', makeReader(false));
  child.stderr?.on('data', makeReader(true));

  const done = new Promise<number>((resolve) => {
    child.on('error', (err) => {
      handlers.onLog?.(`Failed to start Python (${pythonExe()}): ${err.message}`, true);
      resolve(127);
    });
    child.on('close', (code) => resolve(code ?? 1));
  });
  return { child, done };
}

/** Limits (override with env vars). */
export const MAX_UPLOAD_BYTES = Number(process.env.HEIMDALL_MAX_UPLOAD_MB ?? 512) * 1024 * 1024;
export const MAX_CONCURRENT_JOBS = Number(process.env.HEIMDALL_MAX_JOBS ?? 2);
export const JOB_TIMEOUT_MS = Number(process.env.HEIMDALL_JOB_TIMEOUT_S ?? 1800) * 1000;
export const KEEP_JOBS = Number(process.env.HEIMDALL_KEEP_JOBS ?? 50);

let running = 0;
export function acquireSlot(): boolean {
  if (running >= MAX_CONCURRENT_JOBS) return false;
  running += 1;
  return true;
}
export function releaseSlot() {
  running = Math.max(0, running - 1);
}

/** Delete all but the newest KEEP_JOBS job directories (best effort). */
export async function pruneJobs() {
  try {
    const { rm } = await import('fs/promises');
    const jobs = await listJobs(1000);
    for (const j of jobs.slice(KEEP_JOBS)) {
      await rm(jobPath(j.id), { recursive: true, force: true });
    }
  } catch {}
}

export const CONTENT_TYPES: Record<string, string> = {
  '.glb': 'model/gltf-binary',
  '.tif': 'image/tiff',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.json': 'application/json',
  '.bin': 'application/octet-stream',
  '.csv': 'text/csv',
};
