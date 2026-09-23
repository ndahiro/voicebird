import { Building2, GraduationCap, Landmark, Users } from "lucide-react"

export function FeatureInfo() {
    const organizations = [
        {
            icon: GraduationCap,
            name: "Makerere University",
            desc: "Leading research institution"
        },
        {
            icon: Building2,
            name: "SunbirdAI",
            desc: "AI for social impact"
        },
        {
            icon: Landmark,
            name: "Government Agencies",
            desc: "Public sector services"
        },
        {
            icon: Users,
            name: "NGOs & Communities",
            desc: "Grassroots organizations"
        },
    ]

    return (
        <div className="pt-12 pb-8">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold mb-2">Trusted By</h2>
                <p className="text-muted-foreground">
                    Supporting African language accessibility across institutions
                </p>
            </div>

            <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
                {organizations.map((org) => (
                    <div
                        key={org.name}
                        className="group rounded-xl border bg-card/50 p-6 text-center hover:shadow-lg hover:border-primary/50 transition-all duration-300"
                    >
                        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 group-hover:bg-primary/20 transition-colors">
                            <org.icon className="h-7 w-7 text-primary" />
                        </div>
                        <h3 className="font-semibold text-base mb-1">{org.name}</h3>
                        <p className="text-xs text-muted-foreground">{org.desc}</p>
                    </div>
                ))}
            </div>
        </div>
    )
}
