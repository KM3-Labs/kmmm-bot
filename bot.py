import traceback
import discord
import datetime
from discord.ext import commands
import config
from cogs.utils import context

description = """
Fluffy bot
"""

cogs = (
    'cogs.admin',
    'cogs.test',
)


def _prefix_callable(bot, msg):
    user_id = bot.user.id
    base = [f'<@!{user_id}> ', f'<@{user_id}> ']
    return base


class KmmmBot(commands.AutoShardedBot):
    def __init__(self):
        allowed_mentions = discord.AllowedMentions(roles=False, everyone=False, users=True)

        super().__init__(
            command_prefix=_prefix_callable,
            description=description,
            pm_help=None,
            help_attrs=dict(hidden=True),
            chunk_guilds_at_startup=False,
            heartbeat_timeout=150.0,
            allowed_mentions=allowed_mentions,
            enable_debug_events=False
        )

        self.client_id = config.client_id

        for cog in cogs:
            try:
                self.load_extension(cog)
            except Exception:
                print(f'Failed to load cog: {cog}.')
                traceback.print_exc()

    async def on_ready(self):
        if not hasattr(self, 'uptime'):
            self.uptime = datetime.datetime.utcnow()

        print(f'Ready: {self.user} (ID: {self.user.id})')

    async def on_command_error(self, ctx, err):
        pass

    async def process_commands(self, msg):
        ctx = await self.get_context(msg, cls=context.Context)

        if ctx.command is None:
            return

        try:
            await self.invoke(ctx)
        except Exception as e:
            print(f'died: {e}')

    async def on_message(self, msg):
        await self.process_commands(msg)

    async def close(self):
        await super().close()

    def run(self):
        try:
            super().run(config.token, reconnect=True)

        except Exception:
            print('died')

    @property
    def config(self):
        return __import__(config)


if __name__ == '__main__':
    bot = KmmmBot()
    bot.run()
