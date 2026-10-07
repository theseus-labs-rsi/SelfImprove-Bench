import asyncio
import os
import signal

def tool_info():
    return {
        "name": "bash",
        "description": """Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You don't have access to the internet via this tool.
* You do have access to a mirror of common linux and python packages via apt and pip.
* State is persistent across command calls and discussions with the user.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run."
                }
            },
            "required": ["command"]
        }
    }

class BashSession:
    """A session of a bash shell."""
    def __init__(self):
        self._started = False
        self._process = None
        self._timed_out = False
        self._timeout = 120.0  # seconds
        self._sentinel = "<<exit>>"
        self._output_delay = 0.2  # seconds

    async def start(self):
        if self._started:
            return
        self._process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "--noprofile",
            "--norc",
            preexec_fn=os.setsid,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy()  # Ensures inheritance of the current environment
        )
        self._started = True

    async def stop(self):
        if not self._started:
            return
        process = self._process
        if process is not None and process.returncode is None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        transport = getattr(process, "_transport", None)
        if transport is not None:
            transport.close()
        self._process = None
        self._started = False

    async def run(self, command):
        if not self._started:
            raise ValueError("Session has not started.")
        if self._process.returncode is not None:
            raise ValueError(f"Bash has exited with returncode {self._process.returncode}")
        if self._timed_out:
            raise ValueError(
                f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
            )

        # The sentinel must start on a NEW line: `command + "; echo ..."`
        # corrupts a closing heredoc delimiter (PY -> PY; echo ...).
        self._process.stdin.write(
            command.rstrip("\n").encode() + f"\necho '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # Drain both pipes while waiting. Peeking at StreamReader._buffer
        # deadlocks on a large `cat result.json`: the 64-KiB reader buffer
        # fills, pipe backpressure prevents bash from printing the sentinel.
        max_capture = 64 * 1024
        sentinel = self._sentinel.encode()
        captured = bytearray()
        stderr = bytearray()
        pending = b""
        truncated = False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout

        def append_bounded(destination, data):
            nonlocal truncated
            available = max_capture - len(destination)
            destination.extend(data[:max(available, 0)])
            if len(data) > available:
                truncated = True

        async def drain_stderr():
            while True:
                chunk = await self._process.stderr.read(65536)
                if not chunk:
                    break
                append_bounded(stderr, chunk)

        stderr_task = asyncio.create_task(drain_stderr())
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                chunk = await asyncio.wait_for(self._process.stdout.read(65536), remaining)
                if not chunk:
                    raise ValueError("Bash exited before producing the command sentinel")
                pending += chunk
                position = pending.find(sentinel)
                if position >= 0:
                    append_bounded(captured, pending[:position])
                    break
                # Retain enough bytes to detect a sentinel split across reads.
                safe = max(0, len(pending) - len(sentinel) + 1)
                append_bounded(captured, pending[:safe])
                pending = pending[safe:]
            await asyncio.sleep(0)
            output = captured.decode(errors="replace").strip()
            error = stderr.decode(errors="replace").strip()
            if truncated:
                output += "\n[tool output truncated at 65536 bytes; use head/grep for a smaller excerpt]"
            return output, error
        except asyncio.TimeoutError as exc:
            self._timed_out = True
            raise ValueError(
                f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
            ) from exc
        finally:
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)

def filter_error(error):
    # Filter out errors that we do not want to see
    filtered_lines = []
    i = 0
    error_lines = error.splitlines()
    while i < len(error_lines):
        line = error_lines[i]

        # Skip the next lines if ioctl error, add relevant lines
        if "Inappropriate ioctl for device" in line:
            i += 3
            if '<<exit>>' in error_lines[i]:
                i += 1
            while i < len(error_lines) - 1:
                filtered_lines.append(error_lines[i])
                i += 1
            i += 1
            continue

        filtered_lines.append(line)
        i += 1
    return '\n'.join(filtered_lines).strip()

async def tool_function_call(command):
    """Execute a command in the bash shell."""
    bash_session = BashSession()
    try:
        if not bash_session._started:
            await bash_session.start()

        output, error = await bash_session.run(command)
        error = filter_error(error)
        result = ""
        if output:
            result += output
        if error:
            result += "\nError:\n" + error
        return result.strip()
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        await bash_session.stop()

def tool_function(command):
    return asyncio.run(tool_function_call(command))

if __name__ == "__main__":
    # Example usage
    import sys

    # Check if the script is called with arguments
    if len(sys.argv) < 2:
        print("Usage: python bash.py '<command>'")
    else:
        # Extract the command from the command-line arguments
        input_command = ' '.join(sys.argv[1:])
        # Run the tool_function asynchronously
        result = tool_function(input_command)
        print(result)
