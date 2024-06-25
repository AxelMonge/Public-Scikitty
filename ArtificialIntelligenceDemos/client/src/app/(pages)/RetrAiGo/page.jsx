import RAGTryDemo from "@/app/components/RAGComponents/RAGTryDemo";
import TopicHeader from "@/app/components/Topic/TopicHeader";

const DESC = `
    Rule-Based AI with informed search uses predefined rules and heuristics 
    to make decisions and solve problems. It applies informed search strategies 
    to navigate through the solution space efficiently.
`

const RetrAiGo = ({ }) => {
    return (
        <main>
            <article className="text-[#ffffff] pb-10">
                <TopicHeader title="RetrAiGo" description={DESC}/>
                <section className="mx-10 mt-10"> 
                    <h2 className="text-3xl"> Try Demo </h2>
                    <hr className="my-3"/>
                    <RAGTryDemo />
                </section>
            </article>
        </main>
    )
}

export default RetrAiGo;