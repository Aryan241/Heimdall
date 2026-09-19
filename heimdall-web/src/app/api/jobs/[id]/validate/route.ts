import { NextRequest } from 'next/server';
import { readFile, writeFile } from 'fs/promises';
import { JOB_NAME, fileExists, jobPath, runPython, safeExt } from '@/lib/server/heimdall';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/** Validate a job's DSM against an uploaded reference DSM / LiDAR raster (field "reference"). */
export async function POST(req: NextRequest, ctx: RouteContext<'/api/jobs/[id]/validate'>) {
  const { id } = await ctx.params;
  let metaPath: string;
  try {
    metaPath = jobPath(id, `${JOB_NAME}_meta.json`);
  } catch {
    return Response.json({ error: 'invalid job id' }, { status: 400 });
  }
  if (!(await fileExists(metaPath))) return Response.json({ error: 'job not found or not finished' }, { status: 404 });

  const form = await req.formData();
  const ref = form.get('reference');
  if (!(ref instanceof File) || ref.size === 0) return Response.json({ error: 'No reference file' }, { status: 400 });

  const stamp = Date.now().toString(36);
  const refName = `reference_${stamp}${safeExt(ref.name, ['.tif', '.tiff'], '.tif')}`;
  await writeFile(jobPath(id, refName), Buffer.from(await ref.arrayBuffer()));

  const prefix = `validation_${stamp}`;
  const log: string[] = [];
  let reportFile: string | null = null;
  let error: string | null = null;
  const { done } = runPython(
    'validate.py',
    ['--meta', metaPath, '--reference', jobPath(id, refName), '--out-dir', jobPath(id), '--prefix', prefix, '--json-only'],
    {
      onEvent: (e) => {
        if (e.event === 'validation') reportFile = String(e.json);
        if (e.event === 'error') error = String(e.message);
      },
      onLog: (line) => log.push(line),
    },
  );
  const code = await done;
  if (code !== 0 || !reportFile) {
    return Response.json({ error: error ?? log.slice(-5).join('\n') ?? 'validation failed' }, { status: 422 });
  }
  const report = JSON.parse(await readFile(jobPath(id, reportFile), 'utf8'));
  return Response.json({ report });
}
