"use client"

import { Bird, Cloud, Sparkles, LogOut, User } from "lucide-react"
import { useSession, signOut } from "next-auth/react"
import { BackendMode } from "@/lib/types"
import { Button } from "@/components/ui/button"

interface HeaderProps {
    backendMode: BackendMode
}

export function Header({ backendMode }: HeaderProps) {
    const { data: session } = useSession()

    return (
        <header className="relative overflow-hidden border-b bg-card/50 backdrop-blur-sm">
            <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-accent/5 to-primary/5" />
            <div className="container relative mx-auto px-4 py-8">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent shadow-lg">
                            <Bird className="h-7 w-7 text-primary-foreground" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold tracking-tight">VoiceBird</h1>
                            <p className="text-sm text-muted-foreground">African Language Transcription</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            {backendMode === "sunbird" && <><Cloud className="h-4 w-4" /><span>SunbirdAI</span></>}
                            {backendMode === "huggingface" && <><Sparkles className="h-4 w-4" /><span>Hugging Face</span></>}
                            {backendMode === "local" && <><Bird className="h-4 w-4" /><span>Local</span></>}
                        </div>
                        {session && (
                            <div className="flex items-center gap-3">
                                <div className="flex items-center gap-2 text-sm">
                                    <User className="h-4 w-4 text-muted-foreground" />
                                    <span className="font-medium">{session.user?.name || session.user?.email}</span>
                                </div>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => signOut({ callbackUrl: "/login" })}
                                    className="flex items-center gap-2"
                                >
                                    <LogOut className="h-4 w-4" />
                                    Logout
                                </Button>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </header>
    )
}
