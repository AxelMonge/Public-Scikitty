import AboutCard from "@/app/components/AboutCard";
import { ABOUT_US } from "@/app/constants/ABOUT_US";

const AboutUs = ({ }) => {
    return (
        <main>
            <article className="px-10 pt-10 text-white mb-5">
                <h1 className="text-7xl text-center text-blue-400">
                    About Us
                </h1>
                <section className="m-10 flex flex-wrap">
                    {ABOUT_US.membersInfo.map((member, index) => (
                        <AboutCard image={member.image} name={member.name} email={member.email} key={index} />
                    ))}
                </section>
                <section className="text-center">
                    {ABOUT_US.description}
                </section>
            </article>
        </main>
    )
}

export default AboutUs;