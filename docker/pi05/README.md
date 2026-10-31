# pi0.5 images (M1)

LeRobot 0.4.4's `pi05` cannot be constructed in the existing images: it needs the transformers fork its `pi` extra pins
(`transformers.models.siglip.check`), and the images have transformers 4.57.6. Two new images are the existing ones plus
exactly two packages. The existing images are never rebuilt, retagged or removed. X-VLA is not run in these images (the extras
conflict).

| local tag | image id | parent | runs |
|---|---|---|---|
| `vla-pi05:admission` (`admission-pi`) | `sha256:a229f2fbe98f589caaca37068c1ba4a8761d111c6e4855b2360f783c1f290eae` | `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc` (MuJoCo 3.13.0) | pi0.5 training, offline tests |
| `vla-pi05:robosuite` (`robosuite-pi`) | `sha256:2abb18a5290dee46938561e1cdadd506cd3738fa4120482603beb13d647b3e21` | `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` (robosuite 1.5.2, MuJoCo 3.4.0) | pi0.5 rollouts |

Built 2026-10-30. Tags are local handles; commands use the full digest. Each adds 92.6 MB over its parent.

## The two packages (pinned and hash-checked, `requirements-pi.txt`, `--no-deps --require-hashes`)

| package | pin | sha256 of the artifact |
|---|---|---|
| transformers 4.53.3 | `https://github.com/huggingface/transformers/archive/dcddb970176382c0fcf4521b0c0e6fc15894dfe0.zip` (head of `fix/lerobot_openpi`, LeRobot 0.4.4's `pi` extra) | `87a4dc40543efab2d52e5579574169e561ecc29b1fe8d3edadf962df6f062155` |
| tokenizers 0.21.4 | the commit above requires `tokenizers>=0.21,<0.22`; the parents have 0.22.2 | `51b7eabb104f46c1c50b486520555715457ae833d5aee9ff6ae853d1130506ff` (`cp39-abi3-manylinux_2_17_x86_64`) |

If `git ls-remote` later shows another head for that branch, the recorded commit stays. scipy is not installed (neither
pi0 nor pi05 imports it); add it only if a code path is shown to need it, and record it here.

## Build

```bash
DOCKER_BUILDKIT=0 docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy \
  -f docker/pi05/Dockerfile.admission-pi -t vla-pi05:admission docker/pi05
DOCKER_BUILDKIT=0 docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy \
  -f docker/pi05/Dockerfile.robosuite-pi -t vla-pi05:robosuite docker/pi05
```

Classic builder, as for the robosuite image: BuildKit tries to pull the bare `sha256:` base from a registry.
A container that runs code from `/tmp` needs `--tmpfs /tmp:rw,exec` (Docker's tmpfs is `noexec` by default).

## Verification (`experiments/pi05_props_p2/tests/test_image.py`, run in each image)

- `check_whether_transformers_replace_is_installed_correctly()` is true;
- `PI05Policy` constructs on CPU with the P2 features and has 3,616,757,520 parameters;
- `pip check` equals the parent's (clean in `admission-pi`; robosuite's four intended unmet requirements in `robosuite-pi`);
- `pip freeze` equals the parent's freeze (`parent_freeze_<digest>.txt`, captured before the build) except the
  `transformers` and `tokenizers` lines, which are the two pinned ones.

The same test fails in the unmodified parent images (checked 2026-10-30).
