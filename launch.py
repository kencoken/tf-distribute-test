import os
import signal
import locale
import asyncio


class Opts:
    task_count = 2
    log_dir = 'logs'


async def print_to_console(stream, task_name=None):
    while True:
        line = await stream.readline()
        # None output returned when stream closes due to terminating process
        if not line:
            break
        line = line.decode(locale.getpreferredencoding(False)).rstrip()
        with open(os.path.join(Opts.log_dir, '{}.log'.format(task_name.replace(' ', '_'))), 'a') as f:
            f.write('{}\n'.format(line))
        if task_name is not None:
            line = '[{}] {}'.format(task_name, line)
        print(line)


async def create_param_server():
    print('Creating parameter server...')
    proc = await asyncio.create_subprocess_shell("CUDA_VISIBLE_DEVICES= python3 serve.py ps 0",
                                                 stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.PIPE,
                                                 preexec_fn=os.setpgrp)
    proc_name = 'ps'
    try:
        await asyncio.gather(print_to_console(proc.stdout, proc_name), print_to_console(proc.stderr, proc_name))
        retval = await proc.wait()
        print('Parameter server closed with return value: {}'.format(retval))
        return retval
    except asyncio.CancelledError:
        print('Killing parameter server process...')
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)


async def create_worker(task_index, task_count):
    print('Creating worker with task index: {}...'.format(task_index))
    cmd = "CUDA_VISIBLE_DEVICES= python3 serve.py worker {} --task_count {}".format(task_index, task_count)
    proc = await asyncio.create_subprocess_shell(cmd,
                                                 stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.PIPE,
                                                 preexec_fn=os.setpgrp)
    proc_name = 'worker {}'.format(task_index)
    try:
        await asyncio.gather(print_to_console(proc.stdout, proc_name), print_to_console(proc.stderr, proc_name))
        retval = await proc.wait()
        print('Worker closed with return value: {}'.format(retval))
        return retval
    except asyncio.CancelledError:
        print('Killing worker {} process...'.format(task_index))
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)


def main():
    loop = asyncio.get_event_loop()
    coroutines = [create_param_server()] + [create_worker(task_index, Opts.task_count)
                                            for task_index in range(Opts.task_count)]
    tasks = asyncio.gather(*coroutines)
    try:
        loop.run_until_complete(tasks)
        print('Completed!')
    except KeyboardInterrupt:
        tasks.cancel()
        loop.run_until_complete(tasks)
    loop.close()

if __name__ == '__main__':
    main()
