/**
 * Prisma client singleton.
 *
 * Next.js hot-reloads modules in dev, which would spawn a new PrismaClient (and
 * a new DB connection pool) per reload. Caching the instance on globalThis —
 * production keeps a single module-level instance — prevents that exhaustion.
 */
import { PrismaClient } from "@prisma/client"

declare global {
    var prisma: PrismaClient | undefined
}

const client = globalThis.prisma || new PrismaClient({})
if (process.env.NODE_ENV !== "production") globalThis.prisma = client

export default client
