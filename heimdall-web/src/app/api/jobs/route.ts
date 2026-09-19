import { NextRequest } from 'next/server';
import { writeFile } from 'fs/promises';
import {
  JOB_NAME,
  ensureDir,
  jobPath,
  listJobs,
  newJobId,
  runPython,
  safeExt,
  writeJobInfo,
} from '@/lib/server/heimdall';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  return Response.json({ jobs: await listJobs() });
}

/**
 * Create a processing job. multipart/form-data fields:
 *   image (required), reference_dem, gcps (CSV), options (JSON:
 *   { gsd?, bands?, tta?, refine?, autoDem?, mode?: 'head' | 'relative', maxSide? })
 * Responds with a Server-Sent Events stream: job, event, log, done, error.
 */
export async function POST(req: NextRequest) {
  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return Response.json({ error: 'Expected multipart form data' }, { status: 400 });
  }
  const image = form.get('image');
  if (!(image instanceof File) || image.size === 0) {
    return Response.json({ error: 'No image file provided' }, { status: 400 });
  }
  const dem = form.get('reference_dem');
  const gcps = form.get('gcps');
  let options: Record<string, unknown> = {};
  try {
    options = JSON.parse(String(form.get('options') ?? '{}'));
  } catch {}

  const id = newJobId();
  const dir = jobPath(id);
  await ensureDir(dir);

  const imgName = `input${safeExt(image.name, ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'], '.png')}`;
  await writeFile(jobPath(id, imgName), Buffer.from(await image.arrayBuffer()));
  const args = ['--input', jobPath(id, imgName), '--output-dir', dir, '--name', JOB_NAME];

  if (dem instanceof File && dem.size > 0) {
    const demName = `reference_dem${safeExt(dem.name, ['.tif', '.tiff'], '.tif')}`;
    await writeFile(jobPath(id, demName), Buffer.from(await dem.arrayBuffer()));
    args.push('--reference-dem', jobPath(id, demName));
  }
  if (gcps instanceof File && gcps.size > 0) {
    await writeFile(jobPath(id, 'gcps.csv'), Buffer.from(await gcps.arrayBuffer()));
    args.push('--gcps', jobPath(id, 'gcps.csv'));
  }
  const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : null);
  if (num(options.gsd)) args.push('--gsd', String(options.gsd));
  if (num(options.maxSide)) args.push('--max-side', String(Math.round(options.maxSide as number)));
  if (typeof options.bands === 'string' && /^\d(,\d){2}$/.test(options.bands)) args.push('--bands', options.bands);
  if (options.tta === true) args.push('--tta');
  if (options.refine === false) args.push('--no-refine');
  if (options.autoDem === false) args.push('--no-auto-dem');
  if (options.mode === 'relative') args.push('--weights', 'none');

  await writeJobInfo(id, { input: image.name, created: new Date().toISOString(), status: 'running', options });

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      let closed = false;
      const send = (event: string, data: unknown) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
        } catch {
          closed = true;
        }
      };
      send('job', { id, input: image.name });

      const logTail: string[] = [];
      let errorMessage: string | null = null;
      const { child, done } = runPython('infer.py', args, {
        onEvent: (e) => {
          if (e.event === 'error') errorMessage = String(e.message ?? 'Pipeline error');
          send('event', e);
        },
        onLog: (line, isErr) => {
          logTail.push(line);
          if (logTail.length > 300) logTail.shift();
          send('log', { text: line, isErr });
        },
      });

      const abort = () => child.kill('SIGTERM');
      req.signal.addEventListener('abort', abort);

      done.then(async (code) => {
        req.signal.removeEventListener('abort', abort);
        if (code === 0) {
          await writeJobInfo(id, { status: 'done', meta: `${JOB_NAME}_meta.json` });
          send('done', { id, meta: `${JOB_NAME}_meta.json` });
        } else {
          const message = errorMessage ?? (logTail.filter((l) => /error|exception/i.test(l)).pop() ||
            `Pipeline exited with code ${code}`);
          await writeJobInfo(id, { status: 'error', error: message, log: logTail.slice(-60) });
          send('error', { id, message });
        }
        closed = true;
        try {
          controller.close();
        } catch {}
      });
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  });
}
