from discord.ext import commands
from cogs.utils import checks


class Admin(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    @checks.is_owner()
    async def load(self, ctx, *, cog):
        try:
            self.bot.load_extension(cog)
        except commands.ExtensionError as e:
            await ctx.send(f'{e.__class__.name}: {e}')
        else:
            await ctx.send('\N{OK HAND SIGN}')

    @commands.command()
    @checks.is_owner()
    async def unload(self, ctx, *, cog):
        try:
            self.bot.unload_extension(cog)
        except commands.ExtensionError as e:
            await ctx.send(f'{e.__class__.__name__}: {e}')
        else:
            await ctx.send('\N{OK HAND SIGN}')


def setup(bot):
    bot.add_cog(Admin(bot))
