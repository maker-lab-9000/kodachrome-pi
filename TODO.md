# Todo

## GPIO shutter and startup service

- [ ] Confirm the shutter control is a normally-open momentary switch.
- [ ] Choose a BCM GPIO input pin (proposed: BCM 17, physical pin 11).
- [ ] Wire the switch between the GPIO input and ground (proposed: physical pin 9); do not connect it to 5 V.
- [ ] Add an optional `GPIOButtonInput` based on `gpiozero.Button` with the internal pull-up enabled.
- [ ] Add configurable software debounce (proposed default: 50 ms).
- [ ] Trigger one capture immediately on each press and require release before rearming.
- [ ] Keep capture processing in the main thread so GPIO callbacks cannot access the camera concurrently.
- [ ] Ignore or coalesce additional presses while a capture is being processed.
- [ ] Add `--gpio-pin` and `--gpio-bounce` options to `kodachrome-capture`.
- [ ] Bypass the terminal/TTY requirement when GPIO input is selected.
- [ ] Preserve the existing keyboard and preview behavior when GPIO input is not selected.
- [ ] Import GPIO Zero only when GPIO mode is requested so development still works off-Pi.
- [ ] Add GPIO Zero as an optional dependency and document installation with Raspberry Pi OS packages.
- [ ] Add clear errors for an invalid GPIO pin, a missing GPIO dependency, and insufficient GPIO permissions.
- [ ] Handle SIGTERM and SIGINT cleanly, closing both the GPIO input and camera.
- [ ] Add a `deploy/kodachrome-capture.service` systemd unit template using:
  - [ ] `User=george`
  - [ ] `SupplementaryGroups=gpio video`
  - [ ] `/home/george/repos/kodachrome-pi/.venv/bin/kodachrome-capture`
  - [ ] `--no-preview --gpio-pin 17 --gpio-bounce 0.05`
  - [ ] An explicit output directory under `/home/george/Pictures/kodachrome`
  - [ ] A stable `/dev/v4l/by-id/...` camera path when available
  - [ ] `Restart=on-failure`, a short restart delay, and unlimited startup retries
- [ ] Document installing, enabling, stopping, and inspecting the service with `systemctl` and `journalctl`.

### Automated verification

- [ ] Test that GPIO mode works without a TTY.
- [ ] Test that one press produces exactly one capture.
- [ ] Test that holding the switch does not repeat captures.
- [ ] Test that switch bounce does not create duplicate captures.
- [ ] Test that captures cannot overlap.
- [ ] Test that camera errors are reported without ending the input loop.
- [ ] Test that GPIO and camera resources close on normal exit and termination.
- [ ] Test the missing-dependency and invalid-pin error paths.
- [ ] Run the existing keyboard, preview, camera, packaging, and full test suites.

### Raspberry Pi verification

- [ ] Confirm the `george` user can access the `gpio` and `video` groups.
- [ ] Run GPIO mode manually with the fake camera and verify switch behavior.
- [ ] Run GPIO mode manually with the real camera and verify both JPEG outputs and `captures.jsonl`.
- [ ] Install and start the systemd service.
- [ ] Confirm a temporarily unavailable camera causes a retry and eventual recovery.
- [ ] Reboot the Raspberry Pi 3B and verify the service starts automatically.
- [ ] Press the switch after boot and confirm exactly one photo pair is saved per press.
- [ ] Inspect service logs for startup, capture, permission, or camera errors.
