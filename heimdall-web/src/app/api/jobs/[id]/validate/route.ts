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
  if (!(await fileExists(metaPath))) {
    // Most often a stale entry in a browser that has been open while the job directory was
    // cleared, rather than a job that genuinely never finished. Say so, since the fix differs.
    return Response.json(
      { error: 'This job is no longer on disk — it may have been cleared. Reload the page and run it again.' },
      { status: 404 },
    );
  }

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

  // Build a flythrough scene from the reference as well, on the job's own grid, so the two can be
  // compared by eye rather than only by number. It reuses the prediction's grid, texture and
  // vertical origin and runs no inference, so it costs well under a second; a failure here must
  // not lose the validation the user already has.
  let truthScene: string | null = null;
  try {
    const stem = `truth_${stamp}`;
    const { done: truthDone } = runPython(
      'scripts/build_truth_scene.py',
      ['--meta', metaPath, '--reference', jobPath(id, refName), '--out-dir', jobPath(id), '--stem', stem],
      { onEvent: (e) => { if (e.event === 'truth_scene') truthScene = String(e.meta); } },
    );
    if (await truthDone !== 0) truthScene = null;
  } catch {
    truthScene = null;
  }

  return Response.json({ report, truthScene });
}
