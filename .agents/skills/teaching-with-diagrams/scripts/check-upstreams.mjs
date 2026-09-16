#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Directory entries include Git tree hashes, so nested reference changes count.
export function fingerprint(entries) {
  if (!Array.isArray(entries) || !entries.some(e => e.name === 'SKILL.md' && e.type === 'file')) {
    throw new Error('Skill directory missing or no longer contains SKILL.md');
  }
  const rows = entries.map(({ name, type, sha }) => {
    if (typeof name !== 'string' || typeof type !== 'string' || !/^[a-f0-9]{40}$/.test(sha)) {
      throw new Error('Invalid GitHub directory entry');
    }
    return [name, type, sha];
  }).sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
  return createHash('sha256').update(JSON.stringify(rows)).digest('hex');
}

async function github(endpoint) {
  const token = process.env.GH_TOKEN || process.env.GITHUB_TOKEN;
  const response = await fetch(`https://api.github.com/${endpoint}`, {
    headers: { Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    signal: AbortSignal.timeout(20000),
    redirect: 'error',
  });
  if (!response.ok) throw new Error(`GitHub HTTP ${response.status} for ${endpoint}`);
  return response.json();
}

export async function checkSource(source, request = github) {
  const { name, repo, path, commit, fingerprint: baseline, license_sha: licenseSha } = source;
  try {
    if (!/^[\w.-]+\/[\w.-]+$/.test(repo) ||
        !/^[\w./-]+$/.test(path) || path.split('/').some(p => !p || p === '..') ||
        !/^[a-f0-9]{40}$/.test(commit) || !/^[a-f0-9]{64}$/.test(baseline) ||
        !/^[a-f0-9]{40}$/.test(licenseSha)) throw new Error('Invalid upstream baseline');
    const latest = await request(`repos/${repo}/commits/HEAD`);
    if (!/^[a-f0-9]{40}$/.test(latest.sha)) throw new Error('Invalid upstream commit');
    // Resolve once: every read uses the same immutable revision.
    const entries = await request(`repos/${repo}/contents/${path}?ref=${latest.sha}`);
    const license = await request(`repos/${repo}/license?ref=${latest.sha}`);
    if (!/^[a-f0-9]{40}$/.test(license.sha)) throw new Error('Missing license hash');
    const digest = fingerprint(entries);
    const changes = [];
    if (digest !== baseline) changes.push('skill-directory');
    if (license.sha !== licenseSha) changes.push('license');
    return { name, status: changes.length ? 'update-available' : 'current', changes,
      reviewed_commit: commit, latest_commit: latest.sha,
      latest_fingerprint: digest, latest_license_sha: license.sha,
      source: `https://github.com/${repo}/tree/${latest.sha}/${path}`,
      compare: `https://github.com/${repo}/compare/${commit}...${latest.sha}` };
  } catch (error) {
    return { name, status: 'error', error: error.message };
  }
}

export function exitCode(results) {
  return results.some(r => r.status === 'error') ? 1 :
    results.some(r => r.status === 'update-available') ? 2 : 0;
}

export async function main() {
  const lock = JSON.parse(await readFile(new URL('../upstream.lock.json', import.meta.url), 'utf8'));
  if (lock.schema_version !== 1 || !Array.isArray(lock.sources) || !lock.sources.length) {
    throw new Error('Unsupported or empty upstream lock');
  }
  const results = await Promise.all(lock.sources.map(source => checkSource(source)));
  console.log(JSON.stringify({ checked_at: new Date().toISOString(), results }, null, 2));
  process.exitCode = exitCode(results);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    console.error(JSON.stringify({ status: 'error', error: error.message }));
    process.exitCode = 1;
  });
}
